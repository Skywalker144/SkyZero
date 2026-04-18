#!/usr/bin/python3
"""
SkyZero_V4 Data Shuffler.

Shuffles NPZ selfplay data for training, choosing a window size based on a power law.
Adapted from KataGomo-Gom2024's shuffle.py, simplified for SkyZero's data format.

The window size is a power law based on the number of rows N:
  WINDOWSIZE(N) = (N^EXPONENT - MIN_ROWS^EXPONENT) / (EXPONENT * MIN_ROWS^(EXPONENT-1)) * INITIAL_WINDOW_PER_ROW + MIN_ROWS
"""

import sys
import os
import argparse
import math
import time
import zipfile
import shutil
import json
import hashlib
import gc
import multiprocessing
import numpy as np
import _numpy_compat  # noqa: F401 -- installs header-parse compat for legacy npz

KEYS = [
    "encodedInputNCHW",
    "policyTargetsN",
    "opponentPolicyTargetsN",
    "valueTargetsN",
    "sampleWeightsN",
    "policyWeightsN",        # PCR: 0 for cheap-search rows — dropping this from shuffle
    "oppPolicyWeightsN",     # causes cheap rows' noisy policy targets to be trained at full weight
]


def joint_shuffle_take_first_n(n, arrs):
    for arr in arrs:
        assert len(arr) == len(arrs[0])
    perm = np.random.permutation(len(arrs[0]))
    perm = perm[:n]
    return [arr[perm] for arr in arrs]


def get_numpy_npz_headers(filename):
    with zipfile.ZipFile(filename) as z:
        npzheaders = {}
        for subfilename in z.namelist():
            npyfile = z.open(subfilename)
            try:
                np.lib.format.read_magic(npyfile)
            except ValueError:
                print("WARNING: bad file, skipping: %s (bad array %s)" % (filename, subfilename))
                return None
            (shape, is_fortran, dtype) = np.lib.format.read_array_header_1_0(npyfile)
            npzheaders[subfilename] = (shape, is_fortran, dtype)
        return npzheaders


def compute_num_rows(filename):
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile) as e:
        print("WARNING: Cannot read file: %s (%s)" % (filename, e))
        return (filename, None)
    if npheaders is None or len(npheaders) <= 0:
        return (filename, None)

    # Use the first key to determine row count
    for key_name in ["encodedInputNCHW", "encodedInputNCHW.npy"]:
        if key_name in npheaders:
            return (filename, npheaders[key_name][0][0])
    # Fallback: try any key
    for key_name, (shape, _, _) in npheaders.items():
        return (filename, shape[0])
    return (filename, None)


def shardify(input_idx, input_file_group, num_out_files, out_tmp_dirs, keep_prob):
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for _ in range(4)])

    if len(input_file_group) == 0:
        return 0

    arrays_by_key = {k: [] for k in KEYS}
    num_files_not_found = 0

    for input_file in input_file_group:
        try:
            with np.load(input_file) as npz:
                for k in KEYS:
                    arrays_by_key[k].append(npz[k])
        except FileNotFoundError:
            num_files_not_found += 1
            print("WARNING: file not found: ", input_file)

    if len(arrays_by_key[KEYS[0]]) == 0:
        return num_files_not_found

    # Concatenate
    merged = {}
    for k in KEYS:
        merged[k] = np.concatenate(arrays_by_key[k], axis=0) if len(arrays_by_key[k]) > 1 else arrays_by_key[k][0]

    num_rows = merged[KEYS[0]].shape[0]
    num_rows_to_keep = num_rows
    if keep_prob < 1.0:
        num_rows_to_keep = min(num_rows_to_keep, int(round(num_rows_to_keep * keep_prob)))

    # Shuffle and take first n
    arrs = [merged[k] for k in KEYS]
    arrs = joint_shuffle_take_first_n(num_rows_to_keep, arrs)

    # Distribute to output shards
    rand_assts = np.random.randint(num_out_files, size=[num_rows_to_keep])
    counts = np.bincount(rand_assts, minlength=num_out_files)
    countsums = np.cumsum(counts)

    for out_idx in range(num_out_files):
        start = countsums[out_idx] - counts[out_idx]
        stop = countsums[out_idx]
        save_dict = {KEYS[i]: arrs[i][start:stop] for i in range(len(KEYS))}
        np.savez_compressed(
            os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
            **save_dict
        )
    return num_files_not_found


def merge_shards(filename, num_shards_to_merge, out_tmp_dir, batch_size):
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for _ in range(5)])

    arrays_by_key = {k: [] for k in KEYS}

    for input_idx in range(num_shards_to_merge):
        shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
        try:
            with np.load(shard_filename) as npz:
                for k in KEYS:
                    arrays_by_key[k].append(npz[k])
        except FileNotFoundError:
            pass

    if len(arrays_by_key[KEYS[0]]) == 0:
        print("WARNING: empty merge file: ", filename)
        return 0

    merged = {}
    for k in KEYS:
        merged[k] = np.concatenate(arrays_by_key[k], axis=0)

    num_rows = merged[KEYS[0]].shape[0]

    # Shuffle
    arrs = [merged[k] for k in KEYS]
    arrs = joint_shuffle_take_first_n(num_rows, arrs)

    # Truncate to batch_size multiple
    num_batches = num_rows // batch_size
    stop = num_batches * batch_size

    save_dict = {KEYS[i]: arrs[i][:stop] for i in range(len(KEYS))}
    np.savez_compressed(filename, **save_dict)

    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename, "w") as f:
        json.dump({"num_rows": num_rows, "num_batches": num_batches}, f)

    return num_batches * batch_size


class TimeStuff:
    def __init__(self, taskstr):
        self.taskstr = taskstr
    def __enter__(self):
        print("Beginning: %s" % self.taskstr, flush=True)
        self.t0 = time.time()
    def __exit__(self, *args):
        print("Finished: %s in %.1f seconds" % (self.taskstr, time.time() - self.t0), flush=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shuffle SkyZero_V4 selfplay data")
    parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
    parser.add_argument('-min-rows', type=int, default=150000)
    parser.add_argument('-max-rows', type=int, default=None)
    parser.add_argument('-keep-target-rows', type=int, default=2100000)
    parser.add_argument('-expand-window-per-row', type=float, required=True)
    parser.add_argument('-taper-window-exponent', type=float, required=True)
    parser.add_argument('-out-dir', required=True)
    parser.add_argument('-out-tmp-dir', required=True)
    parser.add_argument('-approx-rows-per-out-file', type=int, default=50000)
    parser.add_argument('-num-processes', type=int, required=True)
    parser.add_argument('-batch-size', type=int, required=True)
    parser.add_argument('-worker-group-size', type=int, default=80000)
    parser.add_argument('-only-include-md5-path-prop-lbound', type=float, default=None)
    parser.add_argument('-only-include-md5-path-prop-ubound', type=float, default=None)
    parser.add_argument('-output-npz', action='store_true')
    parser.add_argument('-random-data-dir-pattern', type=str, default='random/tdata/',
                        help='Substring in file paths that identifies random/init selfplay data')

    args = parser.parse_args()

    min_rows = args.min_rows
    max_rows = args.max_rows
    keep_target_rows = args.keep_target_rows
    expand_window_per_row = args.expand_window_per_row
    taper_window_exponent = args.taper_window_exponent
    out_dir = args.out_dir
    out_tmp_dir = args.out_tmp_dir
    approx_rows_per_out_file = args.approx_rows_per_out_file
    num_processes = args.num_processes
    batch_size = args.batch_size
    worker_group_size = args.worker_group_size
    only_include_md5_lbound = args.only_include_md5_path_prop_lbound
    only_include_md5_ubound = args.only_include_md5_path_prop_ubound
    random_data_dir_pattern = args.random_data_dir_pattern

    # Find all NPZ files
    all_files = []
    files_with_unknown_num_rows = []
    with TimeStuff("Finding files"):
        for d in args.dirs:
            for (path, dirnames, filenames) in os.walk(d, followlinks=True):
                for filename in filenames:
                    if not filename.endswith(".npz"):
                        continue
                    full_path = os.path.join(path, filename)
                    files_with_unknown_num_rows.append(full_path)
                    all_files.append((full_path, os.path.getmtime(full_path)))

    print("Total files found: %d" % len(all_files), flush=True)

    # Sort by modification time (oldest first)
    all_files.sort(key=lambda x: x[1])

    # Compute row counts
    with TimeStuff("Computing row counts"):
        with multiprocessing.Pool(num_processes) as pool:
            results = dict(pool.map(compute_num_rows, files_with_unknown_num_rows))
            all_files = [(f, t, results.get(f)) for f, t in all_files]

    # Count total rows, with random data capping (from KataGomo shuffle.py:552-588)
    num_rows_total = 0
    num_random_rows_capped = 0  # Random data rows, capped at min_rows
    num_postrandom_rows = 0     # Non-random rows

    for (filename, mtime, num_rows) in all_files:
        if num_rows is None or num_rows <= 0:
            continue
        num_rows_total += num_rows
        if random_data_dir_pattern and random_data_dir_pattern in filename:
            num_random_rows_capped = min(num_random_rows_capped + num_rows, min_rows)
        else:
            num_postrandom_rows += num_rows

    num_usable_rows = num_random_rows_capped + num_postrandom_rows

    if num_rows_total <= 0:
        print("No rows found")
        sys.exit(0)

    if num_usable_rows < min_rows:
        print("Not enough usable rows: %d < %d (total: %d)" % (num_usable_rows, min_rows, num_rows_total))
        sys.exit(0)

    # Compute window size using power-law formula (using usable rows, not total)
    window_taper_offset = min_rows
    power_law_x = num_usable_rows - min_rows + window_taper_offset
    unscaled = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
    scaled = unscaled / (taper_window_exponent * (window_taper_offset ** (taper_window_exponent - 1)))
    desired_num_rows = int(scaled * expand_window_per_row + min_rows)
    desired_num_rows = max(desired_num_rows, min_rows)
    if max_rows is not None:
        desired_num_rows = min(desired_num_rows, max_rows)

    print("Total rows: %d, Usable rows: %d (random capped: %d, postrandom: %d), Desired window: %d" %
          (num_rows_total, num_usable_rows, num_random_rows_capped, num_postrandom_rows, desired_num_rows), flush=True)

    # Reverse so recent files are first
    all_files.reverse()

    # Select files to fill the window
    desired_input_files = []
    num_rows_used = 0
    for filename, mtime, num_rows in all_files:
        if num_rows is None or num_rows <= 0:
            continue
        desired_input_files.append((filename, num_rows))
        num_rows_used += num_rows
        if num_rows_used >= desired_num_rows:
            break

    print("Using %d files with %d/%d rows" % (len(desired_input_files), num_rows_used, desired_num_rows), flush=True)

    del all_files
    gc.collect()

    np.random.shuffle(desired_input_files)

    # MD5-based filtering for train/val split
    if only_include_md5_lbound is not None or only_include_md5_ubound is not None:
        filtered = []
        for input_file, num_rows_in_file in desired_input_files:
            basename = os.path.basename(input_file)
            hashfloat = int("0x" + hashlib.md5(basename.encode('utf-8')).hexdigest()[:13], 16) / 2**52
            ok = True
            if only_include_md5_lbound is not None and hashfloat < only_include_md5_lbound:
                ok = False
            if only_include_md5_ubound is not None and hashfloat >= only_include_md5_ubound:
                ok = False
            if ok:
                filtered.append((input_file, num_rows_in_file))
        print("MD5 filter: %d/%d files" % (len(filtered), len(desired_input_files)))
        desired_input_files = filtered

    if len(desired_input_files) == 0:
        print("No files after filtering")
        sys.exit(0)

    # Compute keep probability and number of output files
    approx_rows_to_keep = min(num_rows_used, keep_target_rows)
    keep_prob = approx_rows_to_keep / num_rows_used

    num_out_files = max(1, int(round(approx_rows_to_keep / approx_rows_per_out_file)))
    out_files = [os.path.join(out_dir, "data%d.npz" % i) for i in range(num_out_files)]
    out_tmp_dirs = [os.path.join(out_tmp_dir, "tmp.shuf%d" % i) for i in range(num_out_files)]

    print("Writing %d output files, keeping ~%d rows" % (num_out_files, approx_rows_to_keep), flush=True)

    # Clean and create tmp dirs
    for tmp_dir in out_tmp_dirs:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

    # Group input files for parallel processing
    desired_input_file_groups = []
    group_so_far = []
    group_size = 0
    for input_file, num_rows_in_file in desired_input_files:
        if num_rows_in_file <= 0:
            continue
        group_so_far.append(input_file)
        group_size += num_rows_in_file
        if group_size >= worker_group_size:
            desired_input_file_groups.append(group_so_far)
            group_so_far = []
            group_size = 0
    if group_size > 0:
        desired_input_file_groups.append(group_so_far)

    print("Grouped into %d sharding groups" % len(desired_input_file_groups), flush=True)

    if os.path.exists(out_dir):
        raise Exception(out_dir + " already exists")
    os.mkdir(out_dir)

    with multiprocessing.Pool(num_processes) as pool:
        with TimeStuff("Sharding"):
            pool.starmap(shardify, [
                (i, desired_input_file_groups[i], num_out_files, out_tmp_dirs, keep_prob)
                for i in range(len(desired_input_file_groups))
            ])

        with TimeStuff("Merging"):
            num_shards = len(desired_input_file_groups)
            merge_results = pool.starmap(merge_shards, [
                (out_files[i], num_shards, out_tmp_dirs[i], batch_size)
                for i in range(num_out_files)
            ])

    print("Rows by output file:", list(zip(out_files, merge_results)), flush=True)

    # Clean tmp dirs
    for tmp_dir in out_tmp_dirs:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    # Write summary
    with open(out_dir + ".json", 'w') as f:
        json.dump({"total_rows": num_rows_total, "window_rows": desired_num_rows}, f)
