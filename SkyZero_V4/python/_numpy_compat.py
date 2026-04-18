"""Compat patch: tolerate legacy SkyZero npz files that wrote 1-D shapes as `(N)`
instead of the valid Python tuple `(N,)`. numpy 2.x rejects these; numpy 1.x did not.
Import this module once per process before reading such npz files.
"""
import ast
import struct
import numpy as np
import numpy.lib._format_impl as _fi

_PATCHED_ATTR = "_skyzero_compat_patched"

if not getattr(_fi, _PATCHED_ATTR, False):
    def _lenient_read_array_header(fp, version, max_header_size=_fi._MAX_HEADER_SIZE):
        if version == (1, 0):
            hlen = struct.unpack("<H", fp.read(2))[0]
        else:
            hlen = struct.unpack("<I", fp.read(4))[0]
        header_bytes = fp.read(hlen)
        header = header_bytes.decode("utf8" if version[0] >= 3 else "latin1")
        d = ast.literal_eval(header.strip())
        shape = d["shape"]
        if isinstance(shape, int):
            shape = (shape,)
        return shape, d["fortran_order"], _fi.descr_to_dtype(d["descr"])

    _fi._read_array_header = _lenient_read_array_header
    setattr(_fi, _PATCHED_ATTR, True)
