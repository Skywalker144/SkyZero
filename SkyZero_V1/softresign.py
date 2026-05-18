import numpy as np


def softresign_adjust(history, orig_sims, threshold=0.9, lookback=3,
                      min_sims=100, min_weight=0.1):
    """KataGo reduceVisits 的 3-class WDL 版本。

    history 里每个值是过去一步 root MCTS WDL 的 max(W, D, L)。
    max(WDL) 是视角无关的——无论当前玩家在赢、在输还是在和棋,都能正确触发。

    触发条件:最近 lookback 步的 max(WDL) 全部 > threshold。
    缩放:proportion = (window_min - threshold) / (1 - threshold),用 proportion**2
    在 (orig_sims → min_sims) 和 (1.0 → min_weight) 上做插值。

    Returns
    -------
    sims          : int, clamp 到 >= min_sims
    sample_weight : float, ∈ [min_weight, 1.0]
    """
    if len(history) < lookback:
        return orig_sims, 1.0
    window_min = min(history[-lookback:])
    if window_min <= threshold:
        return orig_sims, 1.0
    proportion = (window_min - threshold) / (1.0 - threshold)
    p2 = proportion * proportion
    sims = int(round(orig_sims + p2 * (min_sims - orig_sims)))
    sims = max(sims, min_sims)
    weight = 1.0 + p2 * (min_weight - 1.0)
    return sims, float(weight)


def stochastic_resample(memory, weights):
    """Per-sample stochastic 复制:每个样本以 weight w 进入新池——
    确定性地复制 floor(w) 份,再以概率 (w - floor(w)) 多复制一份。

    weight=1.0  → 恰好一份
    weight=0.1  → 10% 概率一份,90% 概率不进
    weight=2.5  → 两份必进 + 50% 概率第三份

    与 PSW 的 apply_surprise_weighting 同形式,但用 np.random.rand() 而非 randn()。
    """
    out = []
    for sample, w in zip(memory, weights):
        w = float(w)
        if w <= 0.0:
            continue
        whole = int(np.floor(w))
        for _ in range(whole):
            out.append(dict(sample))
        frac = w - whole
        if frac > 0 and np.random.rand() < frac:
            out.append(dict(sample))
    return out
