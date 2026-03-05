import torch
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import traceback
import os
import copy
from tqdm import tqdm
from alphazero import AlphaZero, MCTS, temperature_transform

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

class RemoteModel:
    def __init__(self, rank, request_queue, response_pipe):
        self.rank = rank
        self.request_queue = request_queue
        self.response_pipe = response_pipe
        self.training = False

    def eval(self): self.training = False
    def train(self): self.training = True
    def to(self, device): return self

    def __call__(self, state_tensor):
        state_cpu = state_tensor.detach().cpu()
        self.request_queue.put((self.rank, state_cpu))
        policy_np, value_np = self.response_pipe.recv()
        return torch.tensor(policy_np), torch.tensor(value_np)

def gpu_worker(model_instance, model_state_dict, request_queue, response_pipes, command_queue, args, start_barrier):
    try:
        device = args["device"]
        model = model_instance.to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        start_barrier.wait()
        max_batch_size = len(response_pipes)

        while True:
            try:
                cmd, data = command_queue.get_nowait()
                if cmd == "UPDATE":
                    model.load_state_dict(data)
                    model.eval()
                elif cmd == "STOP":
                    break
            except queue.Empty:
                pass

            batch_states, batch_ranks = [], []
            try:
                rank, state = request_queue.get(timeout=0.01)
                batch_states.append(state)
                batch_ranks.append(rank)
            except queue.Empty:
                continue

            while len(batch_states) < max_batch_size:
                try:
                    rank, state = request_queue.get_nowait()
                    batch_states.append(state)
                    batch_ranks.append(rank)
                except queue.Empty:
                    break

            if not batch_states: continue

            input_tensor = torch.cat(batch_states, dim=0).to(device)
            with torch.no_grad():
                policies, values = model(input_tensor)
            
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

            for i, rank in enumerate(batch_ranks):
                response_pipes[rank].send((policies[i:i+1], values[i:i+1]))

    except Exception as e:
        print(f"GPU Worker crashed: {e}")
        traceback.print_exc()

def selfplay_worker(rank, game, args, request_queue, response_pipe, result_queue, seed, start_barrier):
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        local_args = args.copy()
        local_args["device"] = "cpu"
        remote_model = RemoteModel(rank, request_queue, response_pipe)
        mcts = MCTS(game, local_args, remote_model)

        start_barrier.wait()
        while True:
            memory = []
            state = game.get_initial_state()
            to_play = 1
            while not game.is_terminal(state):
                mcts_policy = mcts.search(state, to_play, args["num_simulations"])
                memory.append({"state": state, "to_play": to_play, "mcts_policy": mcts_policy})
                
                temp = args.get("temperature", 1.0)
                if len(memory) > args.get("temp_threshold", 10): temp = 0.01
                
                action = np.random.choice(len(mcts_policy), p=temperature_transform(mcts_policy, temp))
                state = game.get_next_state(state, action, to_play)
                to_play = -to_play
                
            winner = game.get_winner(state)
            game_data = []
            for sample in memory:
                game_data.append({
                    "encoded_state": game.encode_state(sample["state"], sample["to_play"]),
                    "policy_target": sample["mcts_policy"],
                    "value_target": winner * sample["to_play"]
                })
            result_queue.put((game_data, winner))
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()

class AlphaZeroParallel(AlphaZero):
    def __init__(self, game, model, optimizer, args):
        super().__init__(game, model, optimizer, args)
        self.num_workers = args.get("num_workers", 4)
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.command_queue = mp.Queue()
        self.worker_pipes = [mp.Pipe() for _ in range(self.num_workers)]

    def learn(self):
        start_barrier = mp.Barrier(self.num_workers + 2)
        server_pipes = [p[0] for p in self.worker_pipes]
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        cpu_model_structure = copy.deepcopy(self.model).to("cpu")

        gpu_process = mp.Process(target=gpu_worker, args=(cpu_model_structure, cpu_state_dict, self.request_queue, server_pipes, self.command_queue, self.args, start_barrier))
        gpu_process.start()

        worker_processes = []
        base_seed = int(time.time())
        for i in range(self.num_workers):
            p = mp.Process(target=selfplay_worker, args=(i, self.game, self.args, self.request_queue, self.worker_pipes[i][1], self.result_queue, base_seed + i, start_barrier))
            p.start()
            worker_processes.append(p)

        self.command_queue.put(("UPDATE", {k: v.cpu() for k, v in self.model.state_dict().items()}))
        start_barrier.wait()

        try:
            for i in range(self.args.get("num_iterations", 1000)):
                print(f"Iteration {i+1}")
                games_needed = self.args.get("num_games_per_iter", 10)
                games_collected = 0
                with tqdm(total=games_needed, desc="Collecting Games") as pbar:
                    while games_collected < games_needed:
                        try:
                            game_data, _ = self.result_queue.get(timeout=1.0)
                            self.replay_buffer.add_game(game_data)
                            self.game_count += 1
                            games_collected += 1
                            pbar.update(1)
                        except queue.Empty: continue

                t_loss, p_loss, v_loss = 0, 0, 0
                train_steps = self.args.get("train_steps", 10)
                if self.replay_buffer.is_ready():
                    for _ in tqdm(range(train_steps), desc="Training"):
                        res = self._train_step()
                        if res:
                            t_loss, p_loss, v_loss = t_loss + res[0], p_loss + res[1], v_loss + res[2]
                    
                    self.losses["total_loss"].append(t_loss / train_steps)
                    self.losses["policy_loss"].append(p_loss / train_steps)
                    self.losses["value_loss"].append(v_loss / train_steps)
                    print(f"Loss: {self.losses['total_loss'][-1]:.4f} (P: {self.losses['policy_loss'][-1]:.4f}, V: {self.losses['value_loss'][-1]:.4f})")
                    
                    self.command_queue.put(("UPDATE", {k: v.cpu() for k, v in self.model.state_dict().items()}))

                if (i+1) % self.args.get("save_interval", 10) == 0:
                    self.save_checkpoint(f"checkpoint_parallel_{i+1}.pth")
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.command_queue.put(("STOP", None))
            gpu_process.join()
            for p in worker_processes: p.terminate()
