import torch
import torch.multiprocessing as mp
import numpy as np
import time
from alphazero import AlphaZero, MCTS, temperature_transform
from copy import deepcopy

def play_game_worker(rank, game, model_cls, model_kwargs, model_state_dict, args, seed):
    try:
        # Set seed for this worker
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        device = args['device']
        
        # Instantiate model for this worker
        # We need to make sure we're not creating too many CUDA contexts if unnecessary
        # But user has a good GPU, so we'll try to use it if specified.
        model = model_cls(**model_kwargs).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        mcts = MCTS(game, args, model)
        
        memory = []
        to_play = 1
        state = game.get_initial_state()
        
        def get_randomized_simulations(args):
            base_simulations = args['num_simulations']
            min_ratio = args['playout_cap_min_ratio']
            exponent = args['playout_cap_exponent']
            random_value = np.random.random() ** exponent
            ratio = min_ratio + (1 - min_ratio) * random_value
            return max(1, int(base_simulations * ratio))
        
        while not game.is_terminal(state):
            num_simulations = get_randomized_simulations(args)
            action_probs = mcts.search(state, to_play, num_simulations)
            
            memory.append((state, action_probs, to_play, num_simulations))
            
            if len(memory) >= args['zero_t_step']:
                t = 0.1
            else:
                t = args['temperature']
                
            action = np.random.choice(
                game.action_space_size,
                p=temperature_transform(action_probs, t)
            )
            state = game.get_next_state(state, action, to_play)
            to_play = -to_play
            
        final_state = state
        winner = game.get_winner(final_state)
        
        return_memory = []
        for state, policy_target, to_play, num_sims in memory:
            outcome = winner * to_play
            return_memory.append((
                game.encode_state(state, to_play),
                policy_target,
                outcome,
                num_sims
            ))
            
        return return_memory, winner
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

class AlphaZeroParallel(AlphaZero):
    def __init__(self, game, model, optimizer, args, model_cls=None, model_kwargs=None, num_workers=4):
        super().__init__(game, model, optimizer, args)
        self.num_workers = num_workers
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        
        # If model_cls/kwargs not provided, try to infer or require them
        if self.model_cls is None:
             self.model_cls = type(model)
        if self.model_kwargs is None:
            # This is tricky without explicit args. 
            # We'll assume the user passes them or we fail.
            pass

    def selfplay(self):
        # We override selfplay to run in parallel
        # Note: In the original learn loop, selfplay returns (memory, winner) for ONE game.
        # But we want to run multiple games.
        # So we'll change the learn loop slightly or make this return a list of results.
        # However, to keep learn() mostly unchanged, we might need to handle the batching inside learn().
        # Or, we can make selfplay() return a BATCH of games?
        
        # The original learn loop does:
        # memory, winner = self.selfplay()
        # ... logic for one game ...
        
        # If we change selfplay() to return multiple games, we break the loop structure.
        # So it's better to override learn() entirely.
        pass

    def learn(self):
        # Override learn to support parallel selfplay
        import matplotlib.pyplot as plt
        
        batch_size = self.args['batch_size']
        min_buffer_size = self.args['min_buffer_size']
        train_steps_per_generation = self.args['train_steps_per_generation']
        
        total_train_steps = 0
        recent_game_lens = []
        recent_winners = []
        recent_step_times = [] # Note: this metric might be less accurate in parallel
        recent_train_times = []
        total_start_time = time.time()
        
        last_save_time = time.time()
        savetime_interval = self.args['savetime_interval']
        
        print(f'Buffer Size: {self.replay_buffer.window_size}')
        print(f'Batch Size: {batch_size}')
        print(f'Min Buffer Size: {min_buffer_size}')
        print(f'Train Steps per Generation: {train_steps_per_generation}')
        print(f'Save Time Interval: {savetime_interval}s')
        print(f'Parallel Workers: {self.num_workers}')
        print()
        
        # Determine number of games to play per generation
        # Initially play enough to fill parallel workers
        num_games_to_play = self.num_workers
        
        init_flag = True
        train_game_count = 0
        
        # Setup multiprocessing
        # We use 'spawn' for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        while True:
            self.model.eval()
            selfplay_start = time.time()
            
            # Prepare args for workers
            model_state = self.model.state_dict()
            # To avoid pickling large data repeatedly or issues with CUDA tensors in dict
            # Move state dict to CPU ensures safety
            model_state_cpu = {k: v.cpu() for k, v in model_state.items()}
            
            # Use Pool to run games
            # We create a new pool or reuse? Creating new pool ensures clean state but has overhead.
            # With 'spawn', overhead is non-trivial. 
            # But for board games lasting seconds/minutes, it's okay.
            
            worker_args = []
            base_seed = int(time.time())
            for i in range(self.num_workers):
                worker_args.append((
                    i, 
                    self.game, 
                    self.model_cls, 
                    self.model_kwargs, 
                    model_state_cpu, 
                    self.args, 
                    base_seed + i
                ))
                
            with mp.Pool(processes=self.num_workers) as pool:
                results = pool.starmap(play_game_worker, worker_args)
            
            selfplay_time = time.time() - selfplay_start
            
            # Process results
            # Each result is (memory, winner)
            total_steps_in_batch = 0
            
            for memory, winner in results:
                if not memory: continue # Failed game
                
                self.game_count += 1
                
                recent_winners.append(winner)
                if len(recent_winners) > 100:
                    recent_winners.pop(0)
                
                step_count = len(memory)
                total_steps_in_batch += step_count
                
                recent_game_lens.append(step_count)
                if len(recent_game_lens) > 100:
                    recent_game_lens.pop(0)
                    
                for sample in memory:
                    self.replay_buffer.buffer.append(sample)
            
            avg_game_len = sum(recent_game_lens) / len(recent_game_lens) if recent_game_lens else 0
            recent_first_win = sum(1 for w in recent_winners if w == 1)
            recent_second_win = sum(1 for w in recent_winners if w == -1)
            recent_total = len(recent_winners)
            
            avg_time_per_game_parallel = selfplay_time / self.num_workers
            
            current_buffer_size = len(self.replay_buffer)
            print(f'\n[Game {self.game_count}] (Batch of {self.num_workers}) Total Steps: {total_steps_in_batch}, Buffer: {current_buffer_size}, Avg Len: {avg_game_len:.1f}')
            print(f'  Win Rate (Recent {recent_total}) - First: {recent_first_win}/{recent_total} ({100 * recent_first_win / recent_total:.1f}%), '
                  f'Second: {recent_second_win}/{recent_total} ({100 * recent_second_win / recent_total:.1f}%)')
            print(f'  Selfplay Batch Time: {selfplay_time:.2f}s, Avg/Game (Parallel): {avg_time_per_game_parallel:.2f}s')

            if current_buffer_size < min_buffer_size:
                print(f'  [Skip Training] Buffer {current_buffer_size} < min_buffer_size {min_buffer_size}')
                continue
            elif init_flag:
                train_game_count = self.game_count
                init_flag = False

            current_time = time.time()
            if current_time - last_save_time >= savetime_interval:
                self.save_checkpoint()
                last_save_time = current_time

            # Dynamic adjustment of training frequency based on original logic
            # The original logic calculated num_games_per_generation to keep replay ratio.
            # Here we just check if we have played enough games since last train.
            
            # We trained at train_game_count. Current is self.game_count.
            # We need to see if we should train.
            # But here we already collected a batch. We can just train now.
            # The original code did: run 1 game -> check -> (maybe) train.
            # Here we run N games -> check -> (maybe) train.
            
            # Let's check if we played enough games.
            # Note: We might want to train multiple times if the batch was large?
            # Or just train once per batch but with more steps?
            # Original: train_steps_per_generation steps after 1 game (if conditions met).
            # Wait, original logic:
            # num_games_per_generation = ... (calculated)
            # if self.game_count != train_game_count: continue (wait for more games)
            # else: train(), then update train_game_count += num_games_per_generation
            
            # We can respect this logic.
            if self.game_count < train_game_count:
                print(f'  [Skip Training] Waiting for {train_game_count - self.game_count} more games')
                continue
            
            # If we exceeded train_game_count (likely, since we step by N), we train.
            
            self.model.train()
            train_losses = []
            train_policy_losses = []
            train_value_losses = []

            train_start = time.time()
            
            # We might want to scale up training steps if we played many games?
            # For now keep it simple: train 'train_steps_per_generation'
            
            for step in range(train_steps_per_generation):
                batch = self.replay_buffer.sample(batch_size)
                loss, policy_loss, value_loss = self._train_batch(batch)
                train_losses.append(loss)
                train_policy_losses.append(policy_loss)
                train_value_losses.append(value_loss)
                total_train_steps += 1

            train_time = time.time() - train_start
            
            avg_loss = np.mean(train_losses)
            avg_policy_loss = np.mean(train_policy_losses)
            avg_value_loss = np.mean(train_value_losses)
            self.losses.append(avg_loss)
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            
            print(f'  [Training] {train_steps_per_generation} steps, Avg Loss: {avg_loss:.4f}')
            print(f'  Train Time: {train_time:.2f}s')

            # Update schedule
            # target_ReplayRatio = total_train_samples / total_generated_samples
            # We want train_steps * batch_size / (games * avg_len) ~= target_ratio
            # games = train_steps * batch_size / (avg_len * target_ratio)
            
            calculated_games_per_gen = int(self.args['batch_size'] * self.args['train_steps_per_generation'] / avg_game_len / self.args['target_ReplayRatio'])
            calculated_games_per_gen = max(1, calculated_games_per_gen)
            
            # Update target
            # If we are currently at G, next train is at G + calculated
            # But we might have already overshot.
            train_game_count = max(self.game_count + 1, train_game_count + calculated_games_per_gen)
            
            # If calculated is small (e.g. 2) and we run 10 workers, we will train every batch.
            # If calculated is large (e.g. 50) and we run 10 workers, we will train every 5 batches.
            
            print(f'  Next training after {train_game_count - self.game_count} games (approx {calculated_games_per_gen} target)')
            
            total_elapsed = time.time() - total_start_time
            avg_time_per_game = total_elapsed / self.game_count
            print(f'  Total Elapsed: {total_elapsed / 60:.1f}min, Avg/Game (Total): {avg_time_per_game:.2f}s')

            # Plots
            try:
                plt.figure(figsize=(10, 6))
                plt.yscale('log')
                plt.plot(self.losses)
                plt.title(f'Training Loss (Game {self.game_count})')
                plt.savefig(f"{self.args['file_name']}_losses.png")
                plt.close()

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.set_yscale('log')
                ax1.plot(self.policy_losses, color='blue')
                ax1.set_title(f'Policy Loss')
                ax2.set_yscale('log')
                ax2.plot(self.value_losses, color='orange')
                ax2.set_title(f'Value Loss')
                plt.tight_layout()
                plt.savefig(f"{self.args['file_name']}_losses_detail.png")
                plt.close()
            except Exception as e:
                print(f"Plotting failed: {e}")

