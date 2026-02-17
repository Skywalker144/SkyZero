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
from policy_surprise_weighting import (
    PolicySurpriseWeighter,
    extract_policy_prior_from_root,
)
from utils import print_board

# Set start method to spawn for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
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
        state_tensor: tensor of shape (1, C, H, W) on CPU
        """
        # Optimization: Send CPU tensor directly (uses shared memory) instead of numpy
        # state_tensor is typically (1, C, H, W)
        state_cpu = state_tensor.detach().cpu()

        # Send request: (worker_rank, state_tensor)
        self.request_queue.put((self.rank, state_cpu))

        # Wait for response from dedicated pipe
        # response is (policy_logits_np, value_np)
        policy_np, value_np = self.response_pipe.recv()

        # Convert back to tensor to satisfy MCTS interface
        return torch.tensor(policy_np), torch.tensor(value_np)


def gpu_worker(model_instance, model_state_dict, request_queue, response_pipes, command_queue, args, start_barrier=None):
    """
    The Server process that holds the GPU model and processes batches of requests.
    """
    try:
        device = args['device']
        # print(f"GPU Worker started on {device}")

        # Initialize model
        # model = model_cls(**model_kwargs).to(device)
        model = model_instance.to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        print(f"GPU Worker: Model initialized and weights loaded on {device}")

        # Wait for all workers to be ready before starting
        if start_barrier is not None:
            start_barrier.wait()

        max_batch_size = len(response_pipes)  # Dynamic batch size up to num_workers

        while True:
            # 1. Check for commands (e.g. update weights)
            try:
                cmd, data = command_queue.get_nowait()
                if cmd == 'UPDATE':
                    model.load_state_dict(data)
                    model.eval()
                    print("GPU Worker: Weights updated")
                elif cmd == 'STOP':
                    break
            except queue.Empty:
                pass

            # 2. Collect Batch
            batch_states = []
            batch_ranks = []

            # Greedy Batch Collection (Low Latency)
            # 1. Wait for first item (blocking with small timeout to check commands)
            try:
                if len(batch_states) == 0:
                    rank, state = request_queue.get(timeout=0.01)
                    batch_states.append(state)
                    batch_ranks.append(rank)
            except queue.Empty:
                continue

            # 2. Collect any other immediately available items (Greedy)
            while len(batch_states) < max_batch_size:
                try:
                    rank, state = request_queue.get_nowait()
                    batch_states.append(state)
                    batch_ranks.append(rank)
                except queue.Empty:
                    break

            if not batch_states:
                continue

            # 3. Inference
            # Stack: list of tensors -> (B, C, H, W)
            try:
                # Optimization: Use torch.cat directly on the list of tensors
                # They are already on CPU shared memory, moving to GPU as a batch is efficient
                input_tensor = torch.cat(batch_states, dim=0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    # Handle varying number of outputs (2 or 5)
                    policies, values = outputs[0], outputs[1]

                policies = policies.cpu().numpy()
                values = values.cpu().numpy()

                # 4. Distribute Results
                for i, rank in enumerate(batch_ranks):
                    # policies[i:i+1] keeps the shape (1, Actions)
                    # values[i:i+1] keeps the shape (1, 1)
                    response_pipes[rank].send((policies[i:i + 1], values[i:i + 1]))

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
        local_args['device'] = 'cpu'

        # Create proxy model
        remote_model = RemoteModel(rank, request_queue, response_pipe)

        # Initialize MCTS
        mcts = MCTS(game, local_args, remote_model)

        halflife = np.sqrt(game.board_width * game.board_height)
        move_temperature_init = args['move_temperature_init']
        move_temperature_final = args['move_temperature_final']

        # Wait for all workers to be ready before starting
        if start_barrier is not None:
            start_barrier.wait()

            # Loop to play games continuously
            while True:
                # logic similar to AlphaZero.selfplay() but continuous
                memory = []
                to_play = 1
                state = game.get_initial_state()

                # Soft Resignation Logic
                resign_threshold = args.get('resign_threshold', -0.95)
                soft_resign_playout_prob = args.get('soft_resign_playout_prob', 0.1)
                in_soft_resignation = False

                # Dynamic args fetching could be implemented here if needed
                # For now we use static args passed at start

                def get_randomized_simulations(args):
                    """
                    Playout Cap Randomization: 二选一策略
                    返回 (num_simulations, for_train) 元组
                    """
                    base_simulations = args['num_simulations']
                    fast_simulations = args['fast_simulations']
                    full_search_prob = args.get('full_search_prob', 0.25)

                    for_train = np.random.random() < full_search_prob
                    num_simulations = base_simulations if for_train else fast_simulations
                    return num_simulations, for_train

                while not game.is_terminal(state):
                    num_simulations, for_train = get_randomized_simulations(local_args)

                    if in_soft_resignation:
                        num_simulations = local_args.get('fast_simulations', 50)
                        for_train = True
                        current_sample_weight = 0.01
                    else:
                        current_sample_weight = 1.0

                    # Unpack 3 values (updated MCTS interface)
                    action_probs, policy_prior, root_value = mcts.search(
                        state, to_play, num_simulations, return_policy_prior=True
                    )

                    # Resignation Check
                    if not in_soft_resignation and root_value < resign_threshold:
                        if np.random.random() < soft_resign_playout_prob:
                            in_soft_resignation = True
                            current_sample_weight = 0.01
                        else:
                            # Resign
                            break

                    memory.append((state, action_probs, to_play, for_train, policy_prior, current_sample_weight, root_value))

                    # if len(memory) >= local_args['zero_t_step']:
                    #     t = 0.1
                    # else:
                    #     t = local_args['temperature']

                    current_step = np.count_nonzero(state[-1])
                    max_t = move_temperature_init
                    min_t = move_temperature_final
                    t = min_t + (max_t - min_t) * (0.5 ** (current_step / halflife))

                    action = np.random.choice(
                        game.action_space_size,
                        p=temperature_transform(action_probs, t)
                    )
                    state = game.get_next_state(state, action, to_play)
                    to_play = -to_play

                if game.is_terminal(state):
                    final_state = state
                    winner = game.get_winner(final_state)
                else:
                    # Resigned
                    winner = -to_play
                    final_state = state

                # Process memory
                return_memory = []
                # Short-term Value Targets 预测步数
                stv_steps = [6, 16, 50]
                memory_len = len(memory)

                for i in range(memory_len):
                    state, policy_target, to_play, for_train, policy_prior, sample_weight, root_value = memory[i]
                    outcome = winner * to_play
                    
                    # 计算 Short-term Value Targets
                    stv_targets = []
                    for k in stv_steps:
                        target_idx = i + k
                        if target_idx < memory_len:
                            # 获取未来时刻的 root_value (item 6)
                            # memory[target_idx] = (..., to_play, ..., root_value)
                            next_to_play = memory[target_idx][2]
                            next_root_value = memory[target_idx][6]

                            # 如果 next_to_play == to_play，则视角相同，直接使用
                            # 如果 next_to_play != to_play，则视角相反，取反
                            if next_to_play == to_play:
                                stv_targets.append(next_root_value)
                            else:
                                stv_targets.append(-next_root_value)
                        else:
                            # 游戏结束，使用最终 outcome
                            stv_targets.append(outcome)

                    return_memory.append((
                        game.encode_state(state, to_play),
                        policy_target,
                        outcome,
                        for_train,  # 布尔值：是否用于训练 Policy
                        policy_prior,
                        sample_weight,
                        root_value,
                        stv_targets
                    ))

                # Send result to main process
                result_queue.put((return_memory, winner, final_state))


    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()


class ParallelAlphaZero(AlphaZero):
    def __init__(self, game, model, optimizer, args, num_workers=4):
        # We don't initialize super() immediately completely because we manage MCTS differently
        # But we inherit utility methods
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.num_workers = num_workers

        # Initialize other AlphaZero attributes
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.aux_policy_losses = []
        self.optimistic_losses = []
        self.stv_losses = []
        self.game_count = 0

        # Replay Buffer
        from replay_buffer import ParallelReplayBuffer
        buffer_size = args['buffer_size']
        self.replay_buffer = ParallelReplayBuffer(
            window_size=buffer_size,
            board_size=game.board_size,
        )

        # Initialize Policy Surprise Weighting (PSW)
        self.psw = PolicySurpriseWeighter(
            baseline_weight_ratio=args.get('psw_baseline_ratio', 0.5),
            min_weight=args.get('psw_min_weight', 0.01)
        )

        # Queues and Pipes for Parallel Execution
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.command_queue = mp.Queue()

        # Create a pipe for each worker
        self.worker_pipes = []  # (server_end, client_end)
        for _ in range(num_workers):
            self.worker_pipes.append(mp.Pipe())

    def load_checkpoint(self, filepath=None):
        import glob

        if filepath is None:
            checkpoint_dir = f"{self.args['file_name']}_checkpoints"
            if not os.path.exists(checkpoint_dir):
                print(f"Checkpoint directory not found: {checkpoint_dir}")
                return False

            pattern = os.path.join(checkpoint_dir, "*.pth")
            checkpoint_files = glob.glob(pattern)

            if not checkpoint_files:
                print(f"No checkpoint files found in: {checkpoint_dir}")
                return False

            filepath = max(checkpoint_files, key=os.path.getmtime)
            print(f"Auto-selected latest checkpoint: {filepath}")

        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded")

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Optimizer loaded")

        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']

        if 'policy_losses' in checkpoint:
            self.policy_losses = checkpoint['policy_losses']

        if 'value_losses' in checkpoint:
            self.value_losses = checkpoint['value_losses']

        if 'aux_policy_losses' in checkpoint:
            self.aux_policy_losses = checkpoint['aux_policy_losses']

        if 'optimistic_losses' in checkpoint:
            self.optimistic_losses = checkpoint['optimistic_losses']

        if 'stv_losses' in checkpoint:
            self.stv_losses = checkpoint['stv_losses']

        if 'game_count' in checkpoint:
            self.game_count = checkpoint['game_count']
            print(f"Game count loaded ({self.game_count} games)")

        if 'replay_buffer' in checkpoint:
            self.replay_buffer.load_state(checkpoint['replay_buffer'])
            
            # Check data format compatibility (now requires 7 elements)
            if len(self.replay_buffer) > 0 and len(self.replay_buffer.buffer[0]) != 7:
                 print(f"Warning: Replay buffer format mismatch (expected 7, got {len(self.replay_buffer.buffer[0])}). Clearing buffer.")
                 self.replay_buffer.clear()
            
            print(f"Replay buffer loaded ({len(self.replay_buffer)} samples)")

        print(f"Checkpoint loaded from {filepath}")
        return True

    def learn(self):
        import matplotlib.pyplot as plt

        print(f"Starting Parallel AlphaZero with Batch MCTS")
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
        cpu_model_structure = copy.deepcopy(self.model).to('cpu')

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

        # Force initial weight update to ensure synchronization
        # This fixes the issue where GPU worker might use old weights after checkpoint load
        print("Sending initial weights to GPU Worker...")
        cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.command_queue.put(('UPDATE', cpu_state))

        # Wait for all workers to be ready
        print("Waiting for all workers to be ready...")

        start_barrier.wait()
        print("All workers ready, starting self-play!")

        # 3. Main Training Loop
        try:
            train_game_count = self.game_count
            init_flag = True
            last_save_time = time.time()
            savetime_interval = self.args['savetime_interval']

            recent_game_lens = deque(maxlen=100)
            recent_sample_counts = deque(maxlen=100)
            recent_winners = deque(maxlen=100)
            total_train_steps = 0
            start_time = time.time()

            # Throughput measuring
            perf_start_time = time.time()
            perf_steps_count = 0
            recent_throughput_measurements = deque(maxlen=10)

            # Initial wait for some games
            print("Waiting for games to start...")

            while True:
                # A. Collect Games from Buffer
                new_games = 0
                while not self.result_queue.empty():
                    try:
                        memory, winner, final_state = self.result_queue.get_nowait()
                        self.game_count += 1
                        new_games += 1
                        steps_count = len(memory)
                        perf_steps_count += steps_count

                        weighted_memory, psw_stats = self.psw.process_game(memory)

                        # Remove policy_prior from weighted_memory before adding to buffer
                        # memory format: (encoded_state, policy_target, outcome, for_train, policy_prior, sample_weight, root_value, stv_targets)
                        # desired format: remove policy_prior (idx 4)
                        final_memory = []
                        for sample in weighted_memory:
                            new_sample = list(sample)
                            new_sample.pop(4)
                            final_memory.append(tuple(new_sample))

                        # if psw_stats.get('enabled', False):
                        if self.game_count % 10 == 0:
                            print(f'  [PSW] Ratio: {psw_stats["expansion_ratio"]:.2f}, KL_mean: {psw_stats["kl_mean"]:.4f}, KL_max: {psw_stats["kl_max"]:.4f}')


                        # Update stats
                        recent_winners.append(winner)
                        recent_game_lens.append(len(memory))  # Use original length for stats
                        recent_sample_counts.append(len(final_memory))  # Use filtered length for replay ratio calc

                        # Add to buffer
                        # Use add_game to ensure ring buffer logic is respected for ParallelReplayBuffer
                        self.replay_buffer.add_game(final_memory)

                        # Print info every 10 games
                        if self.game_count % 10 == 0:
                            avg_len = sum(recent_game_lens) / len(recent_game_lens)

                            recent_first_win = sum(1 for w in recent_winners if w == 1)
                            recent_second_win = sum(1 for w in recent_winners if w == -1)
                            recent_draw = sum(1 for w in recent_winners if w == 0)
                            recent_total = len(recent_winners)

                            # Calculate Global Throughput (Sliding Window)
                            current_duration = time.time() - perf_start_time
                            recent_throughput_measurements.append((current_duration, perf_steps_count))

                            total_window_steps = sum(s for _, s in recent_throughput_measurements)
                            total_window_time = sum(t for t, _ in recent_throughput_measurements)
                            global_steps_per_sec = total_window_steps / total_window_time if total_window_time > 0 else 0

                            print(f"\n[Game {self.game_count}] Winner: {int(winner):+d}, Len: {len(memory)}, Buffer: {len(self.replay_buffer)}, AvgLen: {avg_len:.1f}")
                            print(f'  Speed: {global_steps_per_sec:.1f} steps/s')
                            print(
                                f'  Win Rate (Recent {recent_total}) - First: {recent_first_win}/{recent_total} ({100 * recent_first_win / recent_total:.1f}%), '
                                f'Second: {recent_second_win}/{recent_total} ({100 * recent_second_win / recent_total:.1f}%), '
                                f'Draw: {recent_draw}/{recent_total} ({100 * recent_draw / recent_total:.1f}%)'
                            )

                            # Reset performance counters for next interval
                            perf_start_time = time.time()
                            perf_steps_count = 0

                    except queue.Empty:
                        break

                # B. Check if we should train
                # Wait if buffer is too small
                if len(self.replay_buffer) < self.args['min_buffer_size']:
                    if new_games > 0:
                        print(f"Buffer filling: {len(self.replay_buffer)}/{self.args['min_buffer_size']}")
                    time.sleep(1)  # Wait for workers to produce more
                    continue

                # Determine if we should train based on data ratio
                # Simple logic: If we have played X games, train Y steps
                # Or verify against 'train_game_count' schedule like original code

                if self.game_count >= train_game_count:
                    avg_game_len = sum(recent_game_lens) / len(recent_game_lens) if recent_game_lens else 30
                    avg_sample_count = sum(recent_sample_counts) / len(recent_sample_counts) if recent_sample_counts else 30

                    if init_flag:
                        safe_avg_samples = max(1, avg_sample_count)
                        num_games_per_generation = int(self.args['batch_size'] * self.args['train_steps_per_generation'] / safe_avg_samples / self.args['target_ReplayRatio'])
                        train_game_count = self.game_count + num_games_per_generation
                        init_flag = False
                        print(f"  [Init] Skipping immediate training. Next training at game {train_game_count} (+{num_games_per_generation} games)")
                        continue

                    # Train!
                    self.model.train()
                    train_losses = []
                    train_policy_losses = []
                    train_value_losses = []
                    train_aux_policy_losses = []
                    train_optimistic_losses = []
                    train_stv_losses = []

                    # Number of steps
                    steps = self.args['train_steps_per_generation']

                    for _ in range(steps):
                        batch = self.replay_buffer.sample(self.args['batch_size'])
                        loss, p_loss, v_loss, aux_p_loss, opt_loss, stv_loss = self._train_batch(batch)
                        train_losses.append(loss)
                        train_policy_losses.append(p_loss)
                        train_value_losses.append(v_loss)
                        train_aux_policy_losses.append(aux_p_loss)
                        train_optimistic_losses.append(opt_loss)
                        train_stv_losses.append(stv_loss)
                        total_train_steps += 1

                    # Update History
                    self.losses.append(np.mean(train_losses))
                    self.policy_losses.append(np.mean(train_policy_losses))
                    self.value_losses.append(np.mean(train_value_losses))
                    self.aux_policy_losses.append(np.mean(train_aux_policy_losses))
                    self.optimistic_losses.append(np.mean(train_optimistic_losses))
                    self.stv_losses.append(np.mean(train_stv_losses))

                    print(f"\n[Training] Game {self.game_count}, Steps {steps}, Loss: {np.mean(train_losses):.4f}")
                    print(f"  P_Loss: {np.mean(train_policy_losses):.4f}, V_Loss: {np.mean(train_value_losses):.4f}")
                    print(f"  Aux_P: {np.mean(train_aux_policy_losses):.4f}, Opt: {np.mean(train_optimistic_losses):.4f}, STV: {np.mean(train_stv_losses):.4f}")

                    # C. Sync Model with GPU Worker
                    # Send new weights to GPU process
                    # Move to CPU first to avoid CUDA sharing issues in Queue
                    cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    self.command_queue.put(('UPDATE', cpu_state))
                    # print("Main Process: Sent UPDATE command to GPU Worker")

                    # D. Update Schedule
                    target_ratio = self.args['target_ReplayRatio']
                    # new_games_target = steps * batch_size / (avg_len * ratio)
                    # Use avg_sample_count (filtered) to maintain correct ratio with PSW
                    safe_avg_samples = max(1, avg_sample_count)
                    calculated_games = int(self.args['batch_size'] * steps / safe_avg_samples / target_ratio)
                    calculated_games = max(1, calculated_games)

                    train_game_count = self.game_count + calculated_games
                    print(f"Next training after {calculated_games} games")

                    # E. Plotting & Saving
                    current_time = time.time()
                    if current_time - last_save_time >= savetime_interval:
                        self.save_checkpoint()
                        last_save_time = current_time

                    try:
                        # Plotting losses
                        plt.figure(figsize=(10, 6))
                        plt.yscale('log')
                        plt.xlabel('Training Steps (Generations)')
                        plt.ylabel('Loss')
                        plt.plot(self.losses)
                        plt.title(f'Training Loss (Game {self.game_count})')
                        plt.savefig(f"{self.args['file_name']}_losses.png")
                        plt.close()

                        # Plotting detailed losses
                        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
                        
                        # Policy Loss
                        axes[0].set_yscale('log')
                        axes[0].set_ylabel('Policy Loss')
                        axes[0].plot(self.policy_losses, color='blue')
                        axes[0].set_title(f'Policy Loss')
                        axes[0].grid(True, alpha=0.3)
                        
                        # Value Loss
                        axes[1].set_yscale('log')
                        axes[1].set_ylabel('Value Loss')
                        axes[1].plot(self.value_losses, color='orange')
                        axes[1].set_title(f'Value Loss')
                        axes[1].grid(True, alpha=0.3)

                        # Aux Policy Loss
                        axes[2].set_yscale('log')
                        axes[2].set_ylabel('Aux Policy Loss')
                        axes[2].plot(self.aux_policy_losses, color='green')
                        axes[2].set_title(f'Aux Policy Loss')
                        axes[2].grid(True, alpha=0.3)

                        # Optimistic Loss
                        axes[3].set_yscale('log')
                        axes[3].set_ylabel('Optimistic Loss')
                        axes[3].plot(self.optimistic_losses, color='red')
                        axes[3].set_title(f'Optimistic Loss')
                        axes[3].grid(True, alpha=0.3)

                        # STV Loss
                        axes[4].set_yscale('log')
                        axes[4].set_ylabel('STV Loss')
                        axes[4].plot(self.stv_losses, color='purple')
                        axes[4].set_title(f'Short-term Value Loss')
                        axes[4].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(f"{self.args['file_name']}_losses_detail.png")
                        plt.close()
                    except Exception as e:
                        print(f"Plotting failed: {e}")

                else:
                    # Not enough games yet, sleep a tiny bit to let workers work
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            print("Saving checkpoint before exit...")
            self.save_checkpoint()
            # Cleanup
            self.command_queue.put(('STOP', None))
            gpu_process.join()
            for p in worker_processes:
                p.terminate()
