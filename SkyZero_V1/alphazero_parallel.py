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

                if args.get("algo", "puct") == "puct":

                    mcts_policy = mcts.search(state, to_play, args["num_simulations"])
                    t = args.get("move_temperature", 1.0)
                    if len(memory) > args.get("half_life", 10):
                        t = 0.01

                    action = np.random.choice(
                        len(mcts_policy),
                        p=temperature_transform(mcts_policy, t)
                    )

                elif args.get("algo", "puct") == "gumbel":

                    mcts_policy, action, _ = mcts.gumbel_sequential_halving(state, to_play, args["num_simulations"])

                else:
                    raise ValueError(f"unknown algo {args.get('algo')!r}")

                memory.append({"state": state, "to_play": to_play, "mcts_policy": mcts_policy})

                state = game.get_next_state(state, action, to_play)
                to_play = -to_play
                
            winner = game.get_winner(state)
            return_memory = []
            for sample in memory:
                relative_winner = winner * sample["to_play"]
                value_target = np.eye(3, dtype=np.float32)[1 - relative_winner]
                return_memory.append({
                    "encoded_state": game.encode_state(sample["state"], sample["to_play"]),
                    "policy_target": sample["mcts_policy"],
                    "value_target": value_target,
                })
            game_length = np.count_nonzero(state)
            result_queue.put((return_memory, winner, game_length))
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

        needed_games_next_iter = self._compute_needed_games_next_iter()
        self.sps_history.append((time.time(), self.total_samples_selfplay_generated))

        try:
            while True:
                if self.iteration > self.args.get("num_iterations", -1) > 0:
                    print(f"Reached max iterations ({self.args.get('num_iterations', -1)}). Stopping.")
                    break
                if self.total_samples_selfplay_generated >= self.args.get("max_total_samples", -1) > 0:
                    print(f"Reached max total samples ({self.args.get('max_total_samples', -1)}). Stopping.")
                    break

                print(f"\n\n ----- Iteration {self.iteration} -----")
                print(
                    f"TotalGames:{self.total_game_count} "
                    f"TotalSamples:{self.total_samples_selfplay_generated} "
                    f"Window:{self.replay_buffer.window_size()} "
                    f"BufferLen:{len(self.replay_buffer)}"
                )

                # Selfplay
                gw = len(str(needed_games_next_iter))
                game_idx = 0
                while game_idx < needed_games_next_iter:
                    try:
                        memory, winner, game_length = self.result_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    self.replay_buffer.add_game_memory(memory)

                    self.total_samples_selfplay_generated += len(memory)
                    self.total_game_count += 1
                    self.recent_sample_lengths.append(len(memory))
                    self.recent_game_lengths.append(game_length)
                    self.winrate_history.append(winner)

                    if (game_idx + 1) % self.args.get("print_interval", 20) == 0:
                        self.sps_history.append((time.time(), self.total_samples_selfplay_generated))
                        t0, s0 = self.sps_history[0]
                        t1, s1 = self.sps_history[-1]
                        sps = (s1 - s0) / max(t1 - t0, 1e-9)
                        print(
                            f"[SelfPlay] Games={game_idx+1:0{gw}d}/{needed_games_next_iter} "
                            f"Sps={sps:.1f} "
                            f"GameLen:Avg={sum(self.recent_game_lengths)/len(self.recent_game_lengths):.1f} "
                            f"Min={int(min(self.recent_game_lengths))} "
                            f"Max={int(max(self.recent_game_lengths))} "
                            f"Std={int(np.std(self.recent_game_lengths))} "
                            f"BDW={int(self.winrate_history.count(1)/len(self.winrate_history)*100)}/"
                            f"{int(self.winrate_history.count(0)/len(self.winrate_history)*100)}/"
                            f"{int(self.winrate_history.count(-1)/len(self.winrate_history)*100)}",
                            flush=True
                        )
                    game_idx += 1

                # Train
                if self.replay_buffer.is_ready():
                    print()
                    losses = self.train()
                    print(
                        f"[Train] TotalLoss={losses['total_loss']/self.args['train_steps_per_iteration']:.2f} "
                        f"PLoss={losses['policy_loss']/self.args['train_steps_per_iteration']:.2f} "
                        f"VLoss={losses['value_loss']/self.args['train_steps_per_iteration']:.2f}"
                    )
                    self.command_queue.put(("UPDATE", {k: v.cpu() for k, v in self.model.state_dict().items()}))
                    self._record_iter_stats()

                    completed_iter = self.iteration
                    self.iteration += 1
                    if completed_iter % self.args.get("save_interval", 10) == 0:
                        self.save_checkpoint(f"checkpoint_parallel_{completed_iter}.pth")

                needed_games_next_iter = self._compute_needed_games_next_iter()

                self._plot_metrics()

        except KeyboardInterrupt:
            print(f"\nStopping... saving checkpoint at iter={self.iteration}", flush=True)
            if self.iteration > 0:
                self.save_checkpoint(f"checkpoint_parallel_{self.iteration}_interrupt.pth")
        finally:
            self.command_queue.put(("STOP", None))
            gpu_process.join()
            for p in worker_processes: p.terminate()
