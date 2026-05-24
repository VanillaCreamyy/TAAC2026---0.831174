# TAAC 2026 学术赛道初赛方案
本仓库开源的是我们在 TAAC 2026 腾讯广告算法大赛初赛中的最好单模版本：

- 模型版本：`baseline_v9_dualhead_tail_calib`
- 线上 AUC：**0.831174**

该版本整体仍然基于官方 baseline 的 HyFormer / RankMixer 结构，没有完全重写模型。我们的主要思路是在 baseline 主干上补充更有效的数据分布信息，并对高分尾部排序进行轻量校准。

## 一、最终版本实现了什么

### 1. 保留官方 baseline 主干

最终版本没有大幅 scale up 模型，而是保留了官方 baseline 的主体结构：

- HyFormer 主干
- RankMixer NS tokenizer
- sparse embedding
- dense tokenizer
- sequence encoder
- BCE / auxiliary loss 训练框架
- sparse embedding re-init 策略

这样做的原因是，初赛数据规模不大，盲目加深、加宽模型很容易过拟合。我们的实验中，很多更复杂的结构在 valid 上表现不错，但线上并不稳定。

最终版本更偏向“补信息”和“校准排序”，而不是单纯扩大模型。

### 2. Raw dense statistics

我们发现 `user_dense` 特征非常重要，而且分布明显重尾。如果直接把所有 dense 特征压成少量 token，模型会损失一部分尾部分布信息。

因此最终版本加入了 raw dense statistics，用来描述用户 dense 特征的整体分布形态。

主要包括：

- mean / std / max / min
- 非零比例
- 分位数统计
- top-k mass
- high-value mass
- tail count
- chunk concentration
- 稀疏分桶统计

这个方向是最终版本的核心上分点之一。它没有替换原始 dense，而是作为补充信息加入模型，让模型更容易感知用户 dense 的重尾结构。

### 3. EMA checkpoint

最终版本使用了 dense EMA。

EMA 主要用于提升训练后期稳定性。我们观察到某些 epoch 中 raw checkpoint 和 EMA checkpoint 的 valid 表现会有差异，EMA 通常更稳一些。

最终版本在训练中维护 EMA 权重，并使用 EMA checkpoint 作为候选导出权重。

### 4. Online proxy 验证指标

官方 baseline 默认的随机 valid AUC 和线上并不完全一致。我们发现测试集更偏向特定时间段和长历史用户分布，因此最终版本加入了多种验证切片：

- `all_auc`
- `target_auc`
- `hour_auc`
- `online_long_auc`
- `online_proxy_auc`

其中 `online_proxy_auc` 是我们后期用于辅助选择 checkpoint 的指标。它不是线上指标的完美替代，但比只看全量 valid AUC 更稳定。

### 5. Dual-head tail calibration

最终版本最关键的模型结构改动是 **dual-head tail calibration**。

我们观察到 baseline 的整体 AUC 已经不低，但高分尾部样本的排序仍然存在问题。也就是说，模型能大致判断哪些样本更可能转化，但在高置信样本内部，排序仍然不够准。

因此我们加入了一个轻量 tail residual head：

- base head 负责主预测；
- tail head 只对高分尾部做 residual 修正；
- residual scale 控制 tail head 对最终 logit 的影响；
- auxiliary loss 约束 base head，防止 tail head 过拟合；
- 最终输出为 base logit 和 tail residual 的组合。

这个设计的目标不是重建一个新模型，而是在 baseline 已有排序能力上，修正高分尾部的局部排序错误。

## 二、有效 work 点和分数提升

下面是最终版本相关的主要提升路径。分数不是严格单变量消融，因为不同版本之间会同时包含若干改动，且线上存在随机波动，因此仅作为方向参考。

| 版本 / 改动 | 线上 AUC | 相对 baseline 提升 | 说明 |
|---|---:|---:|---|
| 官方 baseline 复跑 | 0.812927 | - | 我们本地复跑后提交的 baseline 分数 |
| 时间 / token 早期优化版本 | 0.823186 | +0.010259 | 说明时间信息和 token 处理对线上有明显帮助 |
| dense stats + EMA 方向 | 0.829781 | +0.016854 | raw dense statistics 是后期最稳定的上分点 |
| dual-head stable 版本 | 0.830541 | +0.017614 | dual-head 校准结构带来进一步提升 |
| 最终版本 `baseline_v9_dualhead_tail_calib` | **0.831174** | **+0.018247** | raw dense stats + EMA + online proxy + tail calibration 的组合版本 |

最终版本相比 baseline 提升约 **+0.0182 AUC**。

## 三、为什么这个版本有效

我们认为最终版本有效，主要来自三点。

### 1. 补回了 user_dense 的重尾信息

`user_dense` 中包含大量重尾分布信息。直接压缩会损失细节，而 raw dense statistics 可以帮助模型理解 dense 特征的整体分布形态。

这类特征对线上比较稳定，是最终版本的基础收益来源。

### 2. checkpoint 选择更贴近线上

只看全量 valid AUC 容易误判。最终版本加入了 online proxy 和多个时间 / 长历史切片指标，帮助我们选择更接近线上分布的 checkpoint。

### 3. 高分尾部排序被轻量修正

AUC 的提升往往来自排序细节。最终版本没有大幅改变主干，而是通过 tail residual head 对高分尾部做轻量校准。

这个改动比较克制，不会像重型 MoE 或 stage2 finetune 那样明显破坏整体排序，因此线上更稳定。

## 四、最终结论

我们的最终方案可以概括为：

> 在官方 baseline 主干上，加入 raw dense statistics 补充用户重尾分布信息，使用 EMA 提升训练稳定性，通过 online proxy 选择更接近线上分布的 checkpoint，并使用 dual-head tail calibration 修正高分尾部排序。

最终单模线上 AUC 为 **0.831174**。
