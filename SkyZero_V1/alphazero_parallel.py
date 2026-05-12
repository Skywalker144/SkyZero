import torch
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import traceback
import os
import copy
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
                
                t = args.get("move_temperature", 1.0)
                if len(memory) > args.get("half_life", 10):
                    t = 0.01
                
                action = np.random.choice(
                    len(mcts_policy),
                    p=temperature_transform(mcts_policy, t)
                )
                state = game.get_next_state(state, action, to_play)
                to_play = -to_play
                
            winner = game.get_winner(state)
            game_data = []
            for sample in memory:
                result = winner * sample["to_play"]
                if result == 1:
                    outcome = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                elif result == -1:
                    outcome = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    outcome = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                game_data.append({
                    "encoded_state": game.encode_state(sample["state"], sample["to_play"]),
                    "policy_target": sample["mcts_policy"],
                    "value_target": outcome,
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
            num_iterations = self.args.get("num_iterations", 1000)
            save_interval = self.args.get("save_interval", 10)
            train_steps_per_iteration = self.args["train_steps_per_iteration"]
            target_replay_ratio = self.args["target_ReplayRatio"]
            unlimited = num_iterations <= 0
            total_label = "inf" if unlimited else str(num_iterations)

            num_next = self._compute_num_next_games()
            i = 0
            it = 0
            while unlimited or i < num_iterations:
                it = i + 1
                print(f"\n=== Iter {it}/{total_label} | "
                      f"collect {num_next} games (RR target={target_replay_ratio}) ===",
                      flush=True)
                print(f"[Stats] total_games={self.game_count} "
                      f"total_samples={self.replay_buffer.total_samples_added}",
                      flush=True)

                # Self-play phase: consume num_next games off the worker queue
                iter_lens, iter_winners = [], []
                sp_t0 = time.time()
                report_every = max(1, num_next // 5)
                games_collected = 0
                while games_collected < num_next:
                    try:
                        game_data, winner = self.result_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    self.replay_buffer.add_game(game_data)
                    self.recent_sample_lengths.append(len(game_data))
                    self.game_count += 1
                    self._record_game(winner, len(game_data))
                    games_collected += 1
                    iter_lens.append(len(game_data))
                    iter_winners.append(winner)
                    if (games_collected % report_every == 0
                            or games_collected == num_next):
                        self._print_selfplay(it, games_collected, num_next, sp_t0,
                                             iter_lens, iter_winners)

                # Training phase
                if self.replay_buffer.is_ready():
                    self._train_iteration(it, train_steps_per_iteration)
                    self.command_queue.put(("UPDATE", {k: v.cpu() for k, v in self.model.state_dict().items()}))
                else:
                    print(f"[Train] iter={it} skipped (buffer not ready: "
                          f"{self.replay_buffer.total_samples_added} < "
                          f"{self.replay_buffer.min_buffer_size})", flush=True)

                num_next = self._compute_num_next_games()

                self.plot_metrics()

                if it % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_parallel_{it}.pth")

                i += 1
        except KeyboardInterrupt:
            print(f"\nStopping... saving checkpoint at iter={it}", flush=True)
            if it > 0:
                self.save_checkpoint(f"checkpoint_parallel_{it}_interrupt.pth")
        finally:
            self.command_queue.put(("STOP", None))
            gpu_process.join()
            for p in worker_processes: p.terminate()
