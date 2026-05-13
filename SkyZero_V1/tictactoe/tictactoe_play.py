import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playgame_web import run_server
from tictactoe_train import train_args


eval_args = {
    "num_simulations": 100,
    "c_puct": 1.5,
    "algo": "puct",
    "host": "127.0.0.1",
    "port": 8765,
    "device": "cuda",
}


if __name__ == "__main__":
    run_server(
        "tictactoe",
        host=eval_args["host"],
        port=eval_args["port"],
        device=eval_args["device"],
        algo=eval_args["algo"],
        sims=eval_args["num_simulations"],
        train_args_override=train_args,
    )
