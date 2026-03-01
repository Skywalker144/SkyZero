import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.onnx
import torch.optim as optim
from alphazero import AlphaZero
from envs.gomoku import Gomoku
from gomoku.gomoku_train import train_args
from nets import ResNet
import onnx

if __name__ == "__main__":

    game = Gomoku(board_size=train_args["board_size"], history_step=train_args["history_step"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(train_args["device"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZero(game, model, optimizer, train_args)
    alphazero.load_model()

    model.eval()
    dummy_input = torch.randn(1, game.num_planes, game.board_size, game.board_size).to(train_args["device"])
    
    onnx_model_name = "web/model.onnx"

    output_names = ["policy_logits", "opponent_policy_logits", "soft_policy_logits", "ownership", "value_logits"]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "opponent_policy_logits": {0: "batch_size"},
            "soft_policy_logits": {0: "batch_size"},
            "ownership": {0: "batch_size"},
            "value_logits": {0: "batch_size"}
        },
        dynamo=False 
    )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")
