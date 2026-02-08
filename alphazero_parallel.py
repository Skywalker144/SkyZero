import torch
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import traceback
from alphazero import AlphaZero, MCTS, temperature_transform
from copy import deepcopy

# Set start method to spawn for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def print_board(board):
    # board shape: (history, size, size)
    # We only care about the last board state (current state)
    current_board = board[-1]
    size = current_board.shape[0]
    
    print("  " + " ".join([f"{i:2}" for i in range(size)]))
    for r in range(size):
        row_str = f"{r:2} "
        for c in range(size):
            val = current_board[r, c]
            if val == 1:
                row_str += " X "  # Black
            elif val == -1:
                row_str += " O "  # White
            else:
                row_str += " . "
        print(row_str)
    print()

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
        # Convert tensor to numpy to strip PyTorch overhead for transmission
        # state_tensor is typically (1, C, H, W)
        state_np = state_tensor.detach().numpy()
        
        # Send request: (worker_rank, state_numpy)
        self.request_queue.put((self.rank, state_np))
        
        # Wait for response from dedicated pipe
        # response is (policy_logits_np, value_np)
        policy_np, value_np = self.response_pipe.recv()
        
        # Convert back to tensor to satisfy MCTS interface
        return torch.tensor(policy_np), torch.tensor(value_np)

def gpu_worker(model_cls, model_kwargs, model_state_dict, request_queue, response_pipes, command_queue, args):
    """
    The Server process that holds the GPU model and processes batches of requests.
    """
    try:
        device = args['device']
        # print(f"GPU Worker started on {device}")
        
        # Initialize model
        model = model_cls(**model_kwargs).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        print(f"GPU Worker: Model initialized and weights loaded on {device}")
        
        max_batch_size = len(response_pipes) # Dynamic batch size up to num_workers
        # Or you can hardcode a limit like 64, 128 etc.
        # max_batch_size = min(len(response_pipes), 64) 
        
        while True:
            # 1. Check for commands (e.g. update weights)
            try:
                if not command_queue.empty():
                    cmd, data = command_queue.get_nowait()
                    if cmd == 'UPDATE':
                        model.load_state_dict(data)
                        model.eval()
                        print("GPU Worker: Weights updated")
                    elif cmd == 'STOP':
                        break
            except Exception:
                pass

            # 2. Collect Batch
            batch_states = []
            batch_ranks = []
            
            # Blocking wait for the first item to avoid busy loop
            try:
                # Timeout allows checking command_queue periodically
                rank, state = request_queue.get(timeout=0.01)
                batch_states.append(state)
                batch_ranks.append(rank)
            except queue.Empty:
                continue
                
            # Collect rest of the batch without waiting too long
            # We want to fill the batch as much as possible but not latency-starve the first item
            start_collect = time.time()
            while len(batch_states) < max_batch_size:
                try:
                    # Very short timeout just to check if data is immediately available
                    rank, state = request_queue.get_nowait()
                    batch_states.append(state)
                    batch_ranks.append(rank)
                except queue.Empty:
                    break
                
                # Safety break to ensure latency isn't too high
                if time.time() - start_collect > 0.001: # 1ms max wait (optimized from 50ms)
                    break
            
            if not batch_states:
                continue
                
            # 3. Inference
            # Stack: list of (1, C, H, W) -> (B, C, H, W)
            # Use concatenate on numpy arrays first
            try:
                input_np = np.concatenate(batch_states, axis=0)
                input_tensor = torch.tensor(input_np, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    policies, values = model(input_tensor)
                
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
                
                # 4. Distribute Results
                for i, rank in enumerate(batch_ranks):
                    # policies[i:i+1] keeps the shape (1, Actions)
                    # values[i:i+1] keeps the shape (1, 1)
                    response_pipes[rank].send((policies[i:i+1], values[i:i+1]))
                    
            except Exception as e:
                print(f"Error in GPU inference: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"GPU Worker crashed: {e}")
        traceback.print_exc()

def selfplay_worker(rank, game, args, request_queue, response_pipe, result_queue, seed):
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
        
        # Loop to play games continuously
        while True:
            # logic similar to AlphaZero.selfplay() but continuous
            memory = []
            to_play = 1
            state = game.get_initial_state()
            
            # Dynamic args fetching could be implemented here if needed
            # For now we use static args passed at start
            
            def get_randomized_simulations(args):
                base_simulations = args['num_simulations']
                min_ratio = args['playout_cap_min_ratio']
                exponent = args['playout_cap_exponent']
                random_value = np.random.random() ** exponent
                ratio = min_ratio + (1 - min_ratio) * random_value
                return max(1, int(base_simulations * ratio))
            
            game_start_time = time.time()
            while not game.is_terminal(state):
                num_simulations = get_randomized_simulations(local_args)
                action_probs = mcts.search(state, to_play, num_simulations)
                
                memory.append((state, action_probs, to_play, num_simulations))
                
                if len(memory) >= local_args['zero_t_step']:
                    t = 0.1
                else:
                    t = local_args['temperature']
                
                action = np.random.choice(
                    game.action_space_size,
                    p=temperature_transform(action_probs, t)
                )
                state = game.get_next_state(state, action, to_play)
                to_play = -to_play
            
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            avg_time_per_step = game_duration / len(memory) if len(memory) > 0 else 0

            final_state = state
            winner = game.get_winner(final_state)
            
            # Process memory
            return_memory = []
            for state, policy_target, to_play, num_sims in memory:
                outcome = winner * to_play
                return_memory.append((
                    game.encode_state(state, to_play),
                    policy_target,
                    outcome,
                    num_sims
                ))
            
            # Send result to main process
            result_queue.put((return_memory, winner, final_state, avg_time_per_step))
            
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()

class AlphaZeroParallel(AlphaZero):
    def __init__(self, game, model, optimizer, args, model_cls=None, model_kwargs=None, num_workers=4):
        # We don't initialize super() immediately completely because we manage MCTS differently
        # But we inherit utility methods
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.num_workers = num_workers
        
        # Initialize other AlphaZero attributes
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.game_count = 0
        
        # Replay Buffer
        from replay_buffer import ReplayBuffer
        buffer_size = args['buffer_size']
        self.replay_buffer = ReplayBuffer(
            window_size=buffer_size,
            board_size=game.board_size,
        )

        if self.model_cls is None:
             self.model_cls = type(model)
        
        # Queues and Pipes for Parallel Execution
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.command_queue = mp.Queue()
        
        # Create a pipe for each worker
        self.worker_pipes = [] # (server_end, client_end)
        for _ in range(num_workers):
            self.worker_pipes.append(mp.Pipe())

    def learn(self):
        import matplotlib.pyplot as plt
        
        print(f"Starting Parallel AlphaZero with Batch MCTS")
        print(f"Workers: {self.num_workers}, Device: {self.args['device']}")
        print(f"Batch Size: {self.args['batch_size']}")
        
        # 1. Start GPU Worker
        server_pipes = [p[0] for p in self.worker_pipes]
        
        # Move state dict to CPU to avoid CUDA pickling issues during spawn
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        gpu_process = mp.Process(
            target=gpu_worker,
            args=(
                self.model_cls, 
                self.model_kwargs, 
                cpu_state_dict,
                self.request_queue,
                server_pipes,
                self.command_queue,
                self.args
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
                    base_seed + i
                )
            )
            p.start()
            worker_processes.append(p)
            
        # 3. Main Training Loop
        try:
            train_game_count = 0
            last_save_time = time.time()
            savetime_interval = self.args['savetime_interval']
            
            recent_game_lens = []
            recent_winners = []
            recent_step_times = []  # Added list for recent step times
            total_train_steps = 0
            start_time = time.time()

            # Throughput measuring
            perf_start_time = time.time()
            perf_steps_count = 0
            recent_throughput_measurements = [] # List of (duration, steps) for sliding window
            
            # Initial wait for some games
            print("Waiting for games to start...")
            
            while True:
                # A. Collect Games from Buffer
                new_games = 0
                while not self.result_queue.empty():
                    try:
                        memory, winner, final_state, avg_time_per_step = self.result_queue.get_nowait()
                        self.game_count += 1
                        new_games += 1
                        steps_count = len(memory)
                        perf_steps_count += steps_count
                        
                        # Update stats
                        recent_winners.append(winner)
                        if len(recent_winners) > 100: recent_winners.pop(0)
                        
                        recent_game_lens.append(len(memory))
                        if len(recent_game_lens) > 100: recent_game_lens.pop(0)

                        recent_step_times.append(avg_time_per_step)
                        if len(recent_step_times) > 100: recent_step_times.pop(0)
                        
                        # Add to buffer
                        for sample in memory:
                            self.replay_buffer.buffer.append(sample)
                            
                        # Optional: Print info occasionally or for special games
                        if self.game_count % 10 == 0:
                            avg_len = sum(recent_game_lens)/len(recent_game_lens)
                            
                            recent_first_win = sum(1 for w in recent_winners if w == 1)
                            recent_second_win = sum(1 for w in recent_winners if w == -1)
                            recent_draw = sum(1 for w in recent_winners if w == 0)
                            recent_total = len(recent_winners)
                            
                            avg_step_time_ms = sum(recent_step_times) / len(recent_step_times) * 1000 if recent_step_times else 0

                            # Calculate Global Throughput (Sliding Window)
                            current_duration = time.time() - perf_start_time
                            
                            # Store (duration, steps) tuples
                            recent_throughput_measurements.append((current_duration, perf_steps_count))
                            if len(recent_throughput_measurements) > 10: # Average over last ~100 games
                                recent_throughput_measurements.pop(0)
                                
                            total_window_steps = sum(s for _, s in recent_throughput_measurements)
                            total_window_time = sum(t for t, _ in recent_throughput_measurements)
                            
                            global_steps_per_sec = total_window_steps / total_window_time if total_window_time > 0 else 0

                            print(f"\n[Game {self.game_count}] Winner: {int(winner):+d}, Len: {len(memory)}, Buffer: {len(self.replay_buffer)}, AvgLen: {avg_len:.1f}")
                            print(f'  Speed: {global_steps_per_sec:.1f} steps/s (Global, Smoothed) | {avg_step_time_ms:.1f} ms/step (Worker Latency)')
                            print(f'  Win Rate (Recent {recent_total}) - First: {recent_first_win}/{recent_total} ({100 * recent_first_win / recent_total:.1f}%), '
                                  f'Second: {recent_second_win}/{recent_total} ({100 * recent_second_win / recent_total:.1f}%), '
                                  f'Draw: {recent_draw}/{recent_total} ({100 * recent_draw / recent_total:.1f}%)')
                            
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
                    time.sleep(1) # Wait for workers to produce more
                    continue
                
                # Determine if we should train based on data ratio
                # Simple logic: If we have played X games, train Y steps
                # Or verify against 'train_game_count' schedule like original code
                
                if self.game_count >= train_game_count:
                    # Train!
                    avg_game_len = sum(recent_game_lens)/len(recent_game_lens) if recent_game_lens else 30
                    
                    self.model.train()
                    train_losses = []
                    train_policy_losses = []
                    train_value_losses = []
                    
                    # Number of steps
                    steps = self.args['train_steps_per_generation']
                    
                    for _ in range(steps):
                        batch = self.replay_buffer.sample(self.args['batch_size'])
                        loss, p_loss, v_loss = self._train_batch(batch)
                        train_losses.append(loss)
                        train_policy_losses.append(p_loss)
                        train_value_losses.append(v_loss)
                        total_train_steps += 1
                        
                    # Update History
                    self.losses.append(np.mean(train_losses))
                    self.policy_losses.append(np.mean(train_policy_losses))
                    self.value_losses.append(np.mean(train_value_losses))
                    
                    print(f"\n[Training] Game {self.game_count}, Steps {steps}, Loss: {np.mean(train_losses):.4f}")
                    
                    # C. Sync Model with GPU Worker
                    # Send new weights to GPU process
                    # Move to CPU first to avoid CUDA sharing issues in Queue
                    cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    self.command_queue.put(('UPDATE', cpu_state))
                    # print("Main Process: Sent UPDATE command to GPU Worker")

                    # D. Update Schedule
                    target_ratio = self.args['target_ReplayRatio']
                    # new_games_target = steps * batch_size / (avg_len * ratio)
                    calculated_games = int(self.args['batch_size'] * steps / avg_game_len / target_ratio)
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
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                        
                        ax1.set_yscale('log')
                        ax1.set_xlabel('Training Steps (Generations)')
                        ax1.set_ylabel('Policy Loss')
                        ax1.plot(self.policy_losses, color='blue')
                        ax1.set_title(f'Policy Loss (Game {self.game_count})')
                        ax1.grid(True, alpha=0.3)
                        
                        ax2.set_yscale('log')
                        ax2.set_xlabel('Training Steps (Generations)')
                        ax2.set_ylabel('Value Loss')
                        ax2.plot(self.value_losses, color='orange')
                        ax2.set_title(f'Value Loss (Game {self.game_count})')
                        ax2.grid(True, alpha=0.3)
                        
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
            # Cleanup
            self.command_queue.put(('STOP', None))
            gpu_process.join()
            for p in worker_processes:
                p.terminate()
