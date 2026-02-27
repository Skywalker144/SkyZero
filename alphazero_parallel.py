import torch
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import traceback
import os
import copy
from collections import deque
from alphazero import AlphaZero, MCTS, temperature_transform
from policy_surprise_weighting import compute_policy_surprise_weights, apply_surprise_weighting_to_game
from utils import print_board

# Set start method to spawn for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class RemoteModel:
    """
    A proxy class that looks like a PyTorch model but sends requests to a prediction queue.
    Used by Self-Play workers to communicate with the GPU worker.
    """

    def __init__(self, rank, request_queue, response_pipe):
        self.rank = rank
        self.request_queue = request_queue
        self.response_pipe = response_pipe
        self.training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def to(self, device):
        # We ignore device movement commands as we run on CPU locally
        return self

    def __call__(self, state_tensor):
        """
        Mimic torch.nn.Module.__call__
        state_tensor: tensor of shape (B, C, H, W) on CPU
        """
        # Send CPU tensor directly (uses shared memory)
        state_cpu = state_tensor.detach().cpu()

        # Send request: (worker_rank, state_tensor)
        self.request_queue.put((self.rank, state_cpu))

        # Wait for response from dedicated pipe
        # response is (policy_logits_np, value_logits_np)
        policy_np, value_np, opponent_policy_np, soft_policy_np, ownership_np = self.response_pipe.recv()

        # Convert back to tensor and return dict to satisfy MCTS interface
        return {
            "policy_logits": torch.tensor(policy_np),
            "value_logits": torch.tensor(value_np),
            "opponent_policy_logits": torch.tensor(opponent_policy_np),
            "soft_policy_logits": torch.tensor(soft_policy_np),
            "ownership": torch.tensor(ownership_np),
        }


def gpu_worker(model_instance, model_state_dict, request_queue, response_pipes, command_queue, args, start_barrier=None):
    """
    The Server process that holds the GPU model and processes batches of requests.
    """
    try:
        device = args["device"]
        # Initialize model
        model = model_instance.to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        print(f"GPU Worker: Model initialized and weights loaded on {device}")

        # Wait for all workers to be ready before starting
        if start_barrier is not None:
            start_barrier.wait()

        max_batch_size = len(response_pipes)

        while True:
            # 1. Check for commands
            try:
                cmd, data = command_queue.get_nowait()
                if cmd == "UPDATE":
                    model.load_state_dict(data)
                    model.eval()
                elif cmd == "STOP":
                    break
            except queue.Empty:
                pass

            # 2. Collect Batch
            batch_states = []
            batch_ranks = []
            batch_sizes = []

            try:
                if len(batch_states) == 0:
                    rank, state = request_queue.get(timeout=0.01)
                    batch_states.append(state)
                    batch_ranks.append(rank)
                    batch_sizes.append(state.size(0))
            except queue.Empty:
                continue

            while len(batch_states) < max_batch_size:
                try:
                    rank, state = request_queue.get_nowait()
                    batch_states.append(state)
                    batch_ranks.append(rank)
                    batch_sizes.append(state.size(0))
                except queue.Empty:
                    break

            if not batch_states:
                continue

            # 3. Inference
            try:
                input_tensor = torch.cat(batch_states, dim=0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    # AlphaZero model returns a dict
                    policies = outputs["policy_logits"]
                    values = outputs["value_logits"]
                    opponent_policies = outputs["opponent_policy_logits"]
                    soft_policies = outputs["soft_policy_logits"]
                    ownership = outputs["ownership"]

                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
                opponent_policies = opponent_policies.cpu().numpy()
                soft_policies = soft_policies.cpu().numpy()
                ownership = ownership.cpu().numpy()

                # 4. Distribute Results
                start_idx = 0
                for i, rank in enumerate(batch_ranks):
                    size = batch_sizes[i]
                    end_idx = start_idx + size
                    response_pipes[rank].send((
                        policies[start_idx:end_idx],
                        values[start_idx:end_idx],
                        opponent_policies[start_idx:end_idx],
                        soft_policies[start_idx:end_idx],
                        ownership[start_idx:end_idx],
                    ))
                    start_idx = end_idx

            except Exception as e:
                print(f"Error in GPU inference: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"GPU Worker crashed: {e}")
        traceback.print_exc()


def selfplay_worker(rank, game, args, request_queue, response_pipe, result_queue, seed, start_barrier=None):
    """
    The Client process that runs the game logic and MCTS.
    """
    try:
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Modify args for CPU execution in this worker
        local_args = args.copy()
        local_args["device"] = "cpu"

        # Create proxy model
        remote_model = RemoteModel(rank, request_queue, response_pipe)

        # Initialize MCTS
        mcts = MCTS(game, local_args, remote_model)

        # Wait for all workers to be ready before starting
        if start_barrier is not None:
            start_barrier.wait()

        # Loop to play games continuously
        while True:
            memory = []
            to_play = 1
            state = game.get_initial_state()

            in_soft_resign = False
            historical_root_value = []

            while not game.is_terminal(state):
                # Get randomized simulations

                if in_soft_resign:
                    num_simulations = args["fast_search_num_simulations"]
                else:
                    if np.random.rand() < args["full_search_prob"]:
                        num_simulations = args["full_search_num_simulations"]
                    else:
                        num_simulations = args["fast_search_num_simulations"]

                mcts_policy, nn_policy, value_probs, root_value = mcts.search(state, to_play, num_simulations)

                # Soft Resign
                historical_root_value.append(root_value)
                absmin_root_value = min(abs(x) for x in historical_root_value[-args.get("soft_resign_step_threshold", 3):])
                if (
                    not in_soft_resign
                    and absmin_root_value >= args.get("soft_resign_threshold", 0.9)
                    and np.random.rand() < args.get("soft_resign_prob", 0.7)
                ):
                    in_soft_resign = True

                if len(memory) > 0:
                    memory[-1]["next_mcts_policy"] = mcts_policy

                memory.append({
                "state": state,
                "to_play": to_play,
                "mcts_policy": mcts_policy,
                "nn_policy": nn_policy,
                "value_probs": value_probs,
                "root_value": root_value,
                "to_play": to_play,
                "is_full_search": num_simulations == args["full_search_num_simulations"],
                "next_mcts_policy": None,
                "sample_weight": 1 if not in_soft_resign else args.get("soft_resign_sample_weight", 0.1),
            })

                # Move temperature decay
                current_step = len(memory)
                max_t = args.get("move_temperature_init", 0.8)
                min_t = args.get("move_temperature_final", 0.2)
                t = min_t + (max_t - min_t) * (0.5 ** (current_step / game.board_size))

                action = np.random.choice(
                    game.board_size**2,
                    p=temperature_transform(mcts_policy, t)
                )
                state = game.get_next_state(state, action, to_play)
                to_play = -to_play

            final_state = state
            winner = game.get_winner(final_state)
            
            return_memory = []
            for sample in memory:
                outcome = winner * sample["to_play"]
                opponent_policy = sample["next_mcts_policy"] if sample["next_mcts_policy"] is not None else np.zeros_like(sample["mcts_policy"])
                sample_data = {
                "final_state": final_state[-1],
                "to_play": sample["to_play"],
                "encoded_state": game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "nn_policy": sample["nn_policy"],
                "value_probs": sample["value_probs"],
                "root_value": sample["root_value"],
                "opponent_policy_target": opponent_policy,
                "outcome": outcome,
                "is_full_search": sample["is_full_search"],
                "sample_weight": sample["sample_weight"],
            }
                return_memory.append(sample_data)
            surprise_weight = compute_policy_surprise_weights(
                return_memory,
                board_size=game.board_size,
                policy_surprise_data_weight=args.get("policy_surprise_data_weight", 0.5),
                value_surprise_data_weight=args.get("value_surprise_data_weight", 0.1),
            )
            return_memory = apply_surprise_weighting_to_game(return_memory, surprise_weight)
            # Send result to main process
            result_queue.put((return_memory, winner, len(memory), final_state))

    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()


class AlphaZeroParallel(AlphaZero):
    def __init__(self, game, model, optimizer, args):
        super().__init__(game, model, optimizer, args)
        self.num_workers = args.get("num_workers", 16)

        # Queues and Pipes for Parallel Execution
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.command_queue = mp.Queue()

        # Create a pipe for each worker
        self.worker_pipes = []  # (server_end, client_end)
        for _ in range(self.num_workers):
            self.worker_pipes.append(mp.Pipe())

    def learn(self):
        print(f"Starting Parallel AlphaZero")
        print(f"Workers: {self.num_workers}, Device: {self.args['device']}")
        print(f"Batch Size: {self.args['batch_size']}")

        # Barrier to synchronize all workers before starting self-play
        # Participants: 1 GPU worker + num_workers self-play workers + 1 main process
        start_barrier = mp.Barrier(self.num_workers + 2)

        # 1. Start GPU Worker
        server_pipes = [p[0] for p in self.worker_pipes]

        # Move state dict to CPU to avoid CUDA pickling issues during spawn
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Create a CPU copy of the model structure to pass to worker
        cpu_model_structure = copy.deepcopy(self.model).to("cpu")

        gpu_process = mp.Process(
            target=gpu_worker,
            args=(
                cpu_model_structure,
                cpu_state_dict,
                self.request_queue,
                server_pipes,
                self.command_queue,
                self.args,
                start_barrier
            )
        )
        gpu_process.start()

        # 2. Start Self-Play Workers
        worker_processes = []
        base_seed = int(time.time())
        for i in range(self.num_workers):
            client_pipe = self.worker_pipes[i][1]
            p = mp.Process(
                target=selfplay_worker,
                args=(
                    i,
                    self.game,
                    self.args,
                    self.request_queue,
                    client_pipe,
                    self.result_queue,
                    base_seed + i,
                    start_barrier
                )
            )
            p.start()
            worker_processes.append(p)

        # Force initial weight update
        cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.command_queue.put(("UPDATE", cpu_state))

        print("Waiting for all workers to be ready...")
        start_barrier.wait()
        print("All workers ready, starting self-play!")

        # 3. Main Training Loop
        try:
            train_game_count = self.game_count
            init_flag = True
            last_save_time = time.time()
            self._last_plot_game_count = self.game_count
            savetime_interval = self.args["savetime_interval"]

            recent_game_lengths = deque(maxlen=self.args.get("len_statistics_queue_size", 300))
            recent_sample_lengths = deque(maxlen=self.args.get("len_statistics_queue_size", 300))
            black_win_counts = deque(maxlen=self.args.get("len_statistics_queue_size", 300))
            white_win_counts = deque(maxlen=self.args.get("len_statistics_queue_size", 300))

            session_start_time = time.time()
            total_samples = 0

            while True:
                # A. Collect Games from Buffer (Limit per iteration to avoid starving the training loop)
                max_games_to_process = 50
                games_processed = 0
                
                while games_processed < max_games_to_process and not self.result_queue.empty():
                    try:
                        memory, winner, game_len, _ = self.result_queue.get_nowait()
                        self.replay_buffer.add_game(memory)

                        self.game_count += 1
                        games_processed += 1
                        total_samples += len(memory)
                        recent_game_lengths.append(game_len)
                        recent_sample_lengths.append(len(memory))

                        black_win_counts.append(1 if winner == 1 else 0)
                        white_win_counts.append(1 if winner == -1 else 0)

                        # Log stats periodically, but do NOT plot here as it is slow
                        if self.game_count % 10 == 0:
                            avg_game_len = np.mean(recent_game_lengths)
                            total_recent = len(black_win_counts)
                            b_rate = np.sum(black_win_counts) / total_recent
                            w_rate = np.sum(white_win_counts) / total_recent
                            d_rate = 1 - b_rate - w_rate

                            self.winrate_history.append((self.game_count, b_rate, w_rate, d_rate))
                            self.avg_game_len_history.append(avg_game_len)

                            elapsed_time = time.time() - session_start_time
                            sps = total_samples / elapsed_time if elapsed_time > 0 else 0

                            print(
                                f"Game: {self.game_count} | Sps: {sps:.1f} | BufferSize: {len(self.replay_buffer)} | "
                                f"WindowSize: {self.replay_buffer.get_window_size()} | "
                                f"AvgGameLen: {avg_game_len:.2f} | BWD: {b_rate:.2f} {w_rate:.2f} {d_rate:.2f}"
                            )

                    except queue.Empty:
                        break

                if self.game_count - self._last_plot_game_count >= self.args.get("plot_interval", 100):
                    self.plot_metrics()
                    self._last_plot_game_count = self.game_count
                # B. Check if we should train
                current_buffer_size = len(self.replay_buffer)
                if current_buffer_size < self.args["min_buffer_size"]:
                    time.sleep(1)
                    continue
                elif init_flag:
                    print("\n" * 5 + "--- Buffer Warmup Complete. Training Started. ---")
                    train_game_count = self.game_count
                    init_flag = False

                current_time = time.time()
                if current_time - last_save_time >= savetime_interval:
                    self.save_checkpoint()
                    # Plot metrics during checkpoint save to avoid frequent slowdowns
                    self.plot_metrics()
                    last_save_time = current_time

                if self.game_count < train_game_count:
                    # Plot every 100 games if not training frequently
                    time.sleep(0.1)
                    continue
                
                print("\n--- Training Session ---")
                # Train!
                self.model.train()
                batch_loss_dict = {key: [] for key in self.losses_dict.keys()}
                full_search_ratio_list = []
                for _ in range(self.args["train_steps_per_generation"]):
                    batch = self.replay_buffer.sample(self.args["batch_size"])
                    loss_dic, full_search_ratio = self._train_batch(batch)
                    for key in batch_loss_dict:
                        batch_loss_dict[key].append(loss_dic[key])
                    full_search_ratio_list.append(full_search_ratio)

                for key in self.losses_dict:
                    self.losses_dict[key].append(np.mean(batch_loss_dict[key]))

                # Sync Model with GPU Worker
                cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                self.command_queue.put(("UPDATE", cpu_state))

                # Update Schedule
                avg_sample_len = np.mean(recent_sample_lengths) if recent_game_lengths else 1
                num_next = int(
                    self.args["batch_size"] * self.args["train_steps_per_generation"] / avg_sample_len / self.args["target_ReplayRatio"]
                )
                num_next = max(1, num_next)
                train_game_count = self.game_count + num_next
                
                print(f"  [Training] Full Search Ratio: {np.mean(full_search_ratio_list):.2f}")
                print(
                    f"  [Training] Loss: {self.losses_dict['total_loss'][-1]:.2f} | "
                    f"Policy Loss: {self.losses_dict['policy_loss'][-1]:.2f} | "
                    f"Value Loss: {self.losses_dict['value_los'][-1]:.2f}"
                )
                print(f"  Next Train after {num_next} games")

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            if self.args.get("save_on_exit", True):
                print("Saving checkpoint before exit...")
                self.save_checkpoint()
            # Cleanup
            self.command_queue.put(("STOP", None))
            gpu_process.join()
            for p in worker_processes:
                p.terminate()
