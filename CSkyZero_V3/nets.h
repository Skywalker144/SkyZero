#ifndef SKYZERO_NETS_H
#define SKYZERO_NETS_H

#include <cmath>
#include <vector>

#include <torch/torch.h>

namespace skyzero {

struct NetworkOutput {
    torch::Tensor policy_logits;
    torch::Tensor opponent_policy_logits;
    torch::Tensor value_logits;
};

struct NormActConvImpl : torch::nn::Module {
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::SiLU act{nullptr};
    torch::nn::Conv2d conv{nullptr};

    NormActConvImpl(int c_in, int c_out, int kernel_size) {
        const int padding = kernel_size / 2;
        bn = register_module("bn", torch::nn::BatchNorm2d(c_in));
        act = register_module("act", torch::nn::SiLU());
        conv = register_module(
            "conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, kernel_size).padding(padding).bias(false))
        );
    }

    torch::Tensor forward(const torch::Tensor& x) {
        return conv->forward(act->forward(bn->forward(x)));
    }
};
TORCH_MODULE(NormActConv);

struct KataGPoolImpl : torch::nn::Module {
    torch::Tensor forward(const torch::Tensor& x) {
        const auto layer_mean = x.mean({2, 3});
        const auto layer_max = torch::amax(x, {2, 3});
        return torch::cat({layer_mean, layer_max}, 1);
    }
};
TORCH_MODULE(KataGPool);

struct ResBlockImpl : torch::nn::Module {
    NormActConv normactconv1{nullptr};
    NormActConv normactconv2{nullptr};

    explicit ResBlockImpl(int channels) {
        normactconv1 = register_module("normactconv1", NormActConv(channels, channels, 3));
        normactconv2 = register_module("normactconv2", NormActConv(channels, channels, 3));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = normactconv1->forward(x);
        out = normactconv2->forward(out);
        return x + out;
    }
};
TORCH_MODULE(ResBlock);

struct GlobalPoolingResidualBlockImpl : torch::nn::Module {
    torch::nn::BatchNorm2d pre_bn{nullptr};
    torch::nn::SiLU pre_act{nullptr};
    torch::nn::Conv2d regular_conv{nullptr};
    torch::nn::Conv2d gpool_conv{nullptr};
    torch::nn::BatchNorm2d gpool_bn{nullptr};
    torch::nn::SiLU gpool_act{nullptr};
    KataGPool gpool{nullptr};
    torch::nn::Linear gpool_to_bias{nullptr};
    NormActConv normactconv2{nullptr};

    GlobalPoolingResidualBlockImpl(int channels, int gpool_channels = -1) {
        if (gpool_channels <= 0) {
            gpool_channels = channels;
        }
        pre_bn = register_module("pre_bn", torch::nn::BatchNorm2d(channels));
        pre_act = register_module("pre_act", torch::nn::SiLU());
        regular_conv = register_module(
            "regular_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false))
        );
        gpool_conv = register_module(
            "gpool_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, gpool_channels, 3).padding(1).bias(false))
        );
        gpool_bn = register_module("gpool_bn", torch::nn::BatchNorm2d(gpool_channels));
        gpool_act = register_module("gpool_act", torch::nn::SiLU());
        gpool = register_module("gpool", KataGPool());
        gpool_to_bias = register_module(
            "gpool_to_bias",
            torch::nn::Linear(torch::nn::LinearOptions(gpool_channels * 2, channels).bias(false))
        );
        normactconv2 = register_module("normactconv2", NormActConv(channels, channels, 3));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = pre_act->forward(pre_bn->forward(x));
        auto regular = regular_conv->forward(out);
        auto g = gpool_act->forward(gpool_bn->forward(gpool_conv->forward(out)));
        auto bias = gpool_to_bias->forward(gpool->forward(g)).unsqueeze(-1).unsqueeze(-1);
        regular = regular + bias;
        regular = normactconv2->forward(regular);
        return x + regular;
    }
};
TORCH_MODULE(GlobalPoolingResidualBlock);

struct NestedBottleneckResBlockImpl : torch::nn::Module {
    NormActConv normactconvp{nullptr};
    torch::nn::ModuleList blockstack;
    NormActConv normactconvq{nullptr};

    NestedBottleneckResBlockImpl(int channels, int mid_channels, int internal_length = 2, bool use_gpool = false) {
        normactconvp = register_module("normactconvp", NormActConv(channels, mid_channels, 1));
        blockstack = register_module("blockstack", torch::nn::ModuleList());
        for (int i = 0; i < internal_length; ++i) {
            if (use_gpool && i == 0) {
                blockstack->push_back(GlobalPoolingResidualBlock(mid_channels));
            } else {
                blockstack->push_back(ResBlock(mid_channels));
            }
        }
        normactconvq = register_module("normactconvq", NormActConv(mid_channels, channels, 1));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = normactconvp->forward(x);
        for (auto& block : *blockstack) {
            if (auto* rb = block->as<ResBlockImpl>()) {
                out = rb->forward(out);
            } else if (auto* gb = block->as<GlobalPoolingResidualBlockImpl>()) {
                out = gb->forward(out);
            }
        }
        out = normactconvq->forward(out);
        return x + out;
    }
};
TORCH_MODULE(NestedBottleneckResBlock);

struct PolicyHeadImpl : torch::nn::Module {
    int board_size;
    torch::nn::Conv2d conv_p{nullptr};
    torch::nn::Conv2d conv_g{nullptr};
    torch::nn::BatchNorm2d g_bn{nullptr};
    torch::nn::SiLU g_act{nullptr};
    KataGPool gpool{nullptr};
    torch::nn::Linear linear_g{nullptr};
    torch::nn::BatchNorm2d p_bn{nullptr};
    torch::nn::SiLU p_act{nullptr};
    torch::nn::Conv2d conv_final{nullptr};

    PolicyHeadImpl(int in_channels, int out_channels, int board_size_, int head_channels = 64)
        : board_size(board_size_) {
        conv_p = register_module("conv_p", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, head_channels, 1).bias(false)));
        conv_g = register_module("conv_g", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, head_channels, 1).bias(false)));
        g_bn = register_module("g_bn", torch::nn::BatchNorm2d(head_channels));
        g_act = register_module("g_act", torch::nn::SiLU());
        gpool = register_module("gpool", KataGPool());
        linear_g = register_module(
            "linear_g",
            torch::nn::Linear(torch::nn::LinearOptions(head_channels * 2, head_channels).bias(false))
        );
        p_bn = register_module("p_bn", torch::nn::BatchNorm2d(head_channels));
        p_act = register_module("p_act", torch::nn::SiLU());
        conv_final = register_module("conv_final", torch::nn::Conv2d(torch::nn::Conv2dOptions(head_channels, out_channels, 1).bias(false)));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto p = conv_p->forward(x);
        auto g = g_act->forward(g_bn->forward(conv_g->forward(x)));
        g = linear_g->forward(gpool->forward(g)).unsqueeze(-1).unsqueeze(-1);
        p = p + g;
        p = p_act->forward(p_bn->forward(p));
        return conv_final->forward(p);
    }
};
TORCH_MODULE(PolicyHead);

struct ValueHeadImpl : torch::nn::Module {
    torch::nn::Conv2d conv_v{nullptr};
    torch::nn::BatchNorm2d v_bn{nullptr};
    torch::nn::SiLU v_act{nullptr};
    KataGPool gpool{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::SiLU act2{nullptr};
    torch::nn::Linear fc_value{nullptr};

    ValueHeadImpl(int in_channels, int out_channels = 3, int head_channels = 32, int value_channels = 64) {
        conv_v = register_module("conv_v", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, head_channels, 1).bias(false)));
        v_bn = register_module("v_bn", torch::nn::BatchNorm2d(head_channels));
        v_act = register_module("v_act", torch::nn::SiLU());
        gpool = register_module("gpool", KataGPool());
        fc1 = register_module("fc1", torch::nn::Linear(head_channels * 2, value_channels));
        act2 = register_module("act2", torch::nn::SiLU());
        fc_value = register_module("fc_value", torch::nn::Linear(value_channels, out_channels));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto v = v_act->forward(v_bn->forward(conv_v->forward(x)));
        auto v_pooled = gpool->forward(v);
        auto out = act2->forward(fc1->forward(v_pooled));
        return fc_value->forward(out);
    }
};
TORCH_MODULE(ValueHead);

struct ResNetImpl : torch::nn::Module {
    int board_size = 0;
    int num_planes = 0;
    int num_blocks = 0;
    int num_channels = 0;

    torch::nn::Sequential start_layer;
    torch::nn::ModuleList trunk_blocks;
    torch::nn::BatchNorm2d trunk_tip_bn{nullptr};
    torch::nn::SiLU trunk_tip_act{nullptr};
    PolicyHead total_policy_head{nullptr};
    ValueHead value_head{nullptr};

    ResNetImpl(int board_size_, int num_planes_, int num_blocks_ = 6, int num_channels_ = 128)
        : board_size(board_size_), num_planes(num_planes_), num_blocks(num_blocks_), num_channels(num_channels_) {
        const int mid_channels = std::max(16, num_channels / 2);

        start_layer = register_module(
            "start_layer",
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(num_planes, num_channels, 3).stride(1).padding(1).bias(false)),
                torch::nn::BatchNorm2d(num_channels),
                torch::nn::SiLU()
            )
        );

        trunk_blocks = register_module("trunk_blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_blocks; ++i) {
            const bool use_gpool = ((i + 2) % 3 == 0);
            trunk_blocks->push_back(NestedBottleneckResBlock(num_channels, mid_channels, 2, use_gpool));
        }
        trunk_tip_bn = register_module("trunk_tip_bn", torch::nn::BatchNorm2d(num_channels));
        trunk_tip_act = register_module("trunk_tip_act", torch::nn::SiLU());

        total_policy_head = register_module("total_policy_head", PolicyHead(num_channels, 2, board_size, num_channels / 2));
        value_head = register_module("value_head", ValueHead(num_channels, 3, num_channels / 4, num_channels / 2));

        init_weights();
    }

    void init_weights() {
        const float silu_gain = std::sqrt(2.35f);
        for (auto& m : modules(false)) {
            if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(m.get())) {
                const auto w = conv->weight;
                const int64_t fan_out = w.size(0) * w.size(2) * w.size(3);
                const float stdv = silu_gain / std::sqrt(static_cast<float>(fan_out));
                torch::nn::init::normal_(conv->weight, 0.0, stdv);
            } else if (auto* bn = dynamic_cast<torch::nn::BatchNorm2dImpl*>(m.get())) {
                torch::nn::init::constant_(bn->weight, 1.0);
                torch::nn::init::constant_(bn->bias, 0.0);
            } else if (auto* lin = dynamic_cast<torch::nn::LinearImpl*>(m.get())) {
                torch::nn::init::normal_(lin->weight, 0.0, 0.01);
            }
        }

        // Fixup-style: scale down the last conv in each residual block so that
        // residual branches start near-zero, stabilizing deep networks
        const float fixup_scale = 1.0f / std::sqrt(static_cast<float>(std::max(num_blocks, 1)));
        for (auto& block : *trunk_blocks) {
            if (auto* nb = block->as<NestedBottleneckResBlockImpl>()) {
                torch::nn::init::normal_(nb->normactconvq->conv->weight, 0.0, fixup_scale * 0.01);
            }
        }
    }

    NetworkOutput forward(const torch::Tensor& x) {
        auto out = start_layer->forward(x);
        for (auto& block : *trunk_blocks) {
            if (auto* nb = block->as<NestedBottleneckResBlockImpl>()) {
                out = nb->forward(out);
            }
        }
        out = trunk_tip_act->forward(trunk_tip_bn->forward(out));

        auto total_policy_logits = total_policy_head->forward(out);
        auto value_logits = value_head->forward(out);

        NetworkOutput output;
        output.policy_logits = total_policy_logits.slice(1, 0, 1);
        output.opponent_policy_logits = total_policy_logits.slice(1, 1, 2);
        output.value_logits = value_logits;
        return output;
    }
};
TORCH_MODULE(ResNet);

}  // namespace skyzero

#endif
