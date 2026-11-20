# Paper-matched Parafoveal Priming v1：实施、记录与首轮结果

> **Recovery 口径修正（2026-09-02）**：早期分析使用 recurrent t0 作为 recovery
> 基线；但 `random_init=true` 且 clean/degraded RNG 同步会使两者在 t0 人为接近。真正的
> 因果基线应是同图片、同 degradation realization、`random_init=false` 的 feedforward
> state。修正后默认 recurrence 在 noise 的 PCoder1–5、occlusion 的 PCoder1–4 均显著
> 优于 feedforward；occlusion PCoder5 均值为正但 CI 跨 0。后续 gain calibration 与
> carrier gate 均使用 feedforward-corrected 指标。第 8–12 节已按该口径更新。共享固定
> 五分类读出也已执行；其终态候选信号及统计边界见第 11 节。

## 1. 本阶段回答什么

本阶段把 Wynn、Bezsudnova 与 Jensen 的电生理论文
[*Neural Sharpening of Object Categories Through Parafoveal Priming*](https://www.biorxiv.org/content/10.1101/2025.05.16.654463v1.abstract)
转成可运行的 Predify/Hebbian VGG16 协议，并把两个机制诊断放进同一条证据链：

1. paper-matched clean no-prime / 1-back / 2-back，检验隐藏表征是否同时出现
   repetition suppression 与 category decoding enhancement；
2. target-only clean-reference diagnostic，区分“网络内部误差变化”和“受损信息真的向
   clean 表征恢复”；
3. Same/Different prime × memory carrier 的因果 pilot，实现并冒烟验证，但只在 recovery
   门槛通过后才扩为正式 200-target 实验；
4. 在另一批 200 张 target-only 图像上训练共同五分类读出，并冻结同一组权重比较全部
   Same/Different carrier 条件，检验 hidden effect 能否被固定下游决策读取。

当前结论是：**模型已在 clean 与 degraded 条件中产生时间局限的论文样
suppression + decoding enhancement。Same prime 也显著改善 PCoder4/5 到 clean
reference 的恢复；Hebbian carry 提供持续而较大的对象特异 recovery，PCoder carry 提供
较早但会衰减的状态优势。原始 VGG 1000-way Top-1/Top-5 仍处于 floor。固定五分类读出
在 noise、联合 carrier、FC2 t9 出现 +2.5 pp 的候选终态收益；bootstrap CI 排除 0，
但精确 McNemar 双侧 p=0.0625，且该点尚未在新数据上确认，因此不能升级为确定性结论。**

## 2. Paper-matched 刺激和时间结构

每个 trial 使用两个连续的 parafoveal displays，再呈现中央 target：

```text
preview 1：左/右各一个物体，2 recurrent steps
preview 2：左/右各一个物体，2 recurrent steps
target：中央一个物体，10 recurrent steps
```

三个 clean 条件只改变同一 target 是否出现在 previews 中：

| condition | target 重复位置 | 实现含义 |
|---|---:|---|
| `no_prime` | 不出现 | 四个 preview 物体都不是 target |
| `one_back` | preview 2 | target 的完全相同 ImageNet 图像替换 preview 2 的一个位置 |
| `two_back` | preview 1 | target 图像替换 preview 1 的一个位置，preview 2 形成干扰间隔 |

左右位置按 target 顺序平衡。Same prime 是像素级同一 target 样本；Different prime 使用原
slot 中来自另一 supercategory 的确定性 distractor。所有物体先保持原 ImageNet transform，
再缩放到 224×224 灰背景 canvas 的 15%；parafoveal 中心横向偏移为画布宽度的 17.5%。
该路径不执行旧 half-field 实验中的 contrast reduction、Gaussian blur 或 downsample。

目标集合为 200 张，五类各 40 张：animals、clothing、food、plants、vehicles。ImageNet
类别被映射到这五个 supercategories；抽样在类别内展开，并使用原 ImageNet class 作为
object identity。分析时采用 `StratifiedGroupKFold`，同一 ImageNet identity 不会同时进入
SVM 的训练折和测试折。

## 3. 模型状态与四种 memory carrier

模型使用 torchvision `VGG16_Weights.IMAGENET1K_V1` 的原始 FC、仓库内五组预训练
PCoder、`random_init=true`。因此 target t0 是第一个 predictive state，不是纯
feedforward VGG state。

paper clean benchmark 的 active 条件只允许 Hebbian state 从 previews 传到 target：

- 每个 preview 完成一次 deferred Hebbian commit；
- target 不再 commit，避免把“target 自身学习”误当作 priming；
- 每个新 target trial 开始前清空 Hebbian state；
- no-Hebb 控制保留相同 PCoder、输入、随机种子和时间步，但旁路 Hebbian modules。

正式因果实验实现四种 carrier：

| carrier | Hebbian carry | PCoder recurrent-state carry |
|---|---:|---:|
| `none` | 否 | 否 |
| `hebb_only` | 是 | 否 |
| `pcoder_carry` | 否 | 是 |
| `hebb_and_pcoder` | 是 | 是 |

`pcoder_carry` 不再调用会自动 reset 的 `model(x)`；它替换 `input_mem` 后调用
`model(None)`，保留各 PCoder 的 representation/prediction/gradient memory。正式
Same−Different 比较不会使用 `none` 作为对象特异效应，因为没有跨 trial memory 时两者
理论上应相同；`none` 每个 degradation 只保存一次，作为计算基线。

## 4. Clean-reference recovery 的定义

诊断条件关闭跨 trial memory，只给模型 noise 0.5 或 fractal occlusion 0.3 的中央 target。
另一个参数完全相同的 reference model 同步处理 clean target。两者使用同步的 fresh random
initialization，避免 `random_init=true` 被误当作 recovery 差异。

每个样本、每个 target timestep、每个 PCoder 均记录：

- degraded representation 到 clean representation；
- degraded downward prediction 到 clean downward prediction；
- PCoder 内部 prediction error；
- cosine、normalized L2 与 MSE。

PCoder1 prediction 对应 clean input；PCoder2–5 prediction 对应同一 timestep 的 clean
reference 下一级 representation。`recovery_change_t0_tlast.csv` 同时保存原始 t9−t0 和
方向统一后的 `oriented_improvement`：cosine 上升为正，distance/MSE 下降为正。

这一区分非常重要：prediction 更像 clean 只说明生成支路产生了更合理的向下预测；只有
representation 也更像 clean，才说明误差校正真正把信息写回了分类所消费的状态。

## 5. 保存 schema

正式 `.pckl` 使用通用 `schema_version=4` 与 `paper_schema_version=1`。主要字段如下：

| field | shape / type | 说明 |
|---|---|---|
| `run_metadata` | dict | DOI、UTC 时间、config snapshot、seed、device、protocol、condition、identity、carrier、degradation、Hebbian 状态 |
| `trial_metadata` | list[dict] | target dataset index、ImageNet label、supercategory、preview slot/side/display、Same/Different 对应 index |
| `dataset_indices` | `[N]` | target 在 ImageNet validation dataset 中的 index |
| `imagenet_labels` | `[N]` | 1000 类 ground-truth |
| `category_labels` | `[N]` | 五个 paper supercategories 的 0–4 标签 |
| `object_identity_labels` | `[N]` | 用于 grouped CV 的 ImageNet class identity |
| `target_logits` | `[N,10,1000]`, float32 | 每个 target timestep 的 logits |
| `target_top1_correct` / `target_top5_correct` | `[N,10]`, bool | 逐样本正确性，可做配对 bootstrap |
| `target_top1_accuracy_timeline` / `target_top5_accuracy_timeline` | `[10]` | 汇总准确率 |
| `target_representations.conv4/conv5` | `[N,10,512]`, float16 | global-average-pooled conv state |
| `target_representations.fc1/fc2` | `[N,10,4096]`, float16 | FC hidden state |
| `representation_recovery` | `[N,10,5,3]`, float32 或 null | PCoder1–5 × cosine/normalized L2/MSE |
| `prediction_recovery` | `[N,10,5,3]`, float32 或 null | downward prediction 的 clean-reference metrics |
| `pcoder_internal_error` | `[N,10,5]`, float32 或 null | 每个 PCoder 内部 error |

首批 formal clean 文件生成时，`trial_metadata.preview_indices` 保存的是四个基础 distractor；
实际 Same prime 可由 `prime_slot` 与 `prime_dataset_index` 无歧义重建。当前 runner 已将记录
明确拆成 `base_distractor_indices` 和 `actual_preview_indices`，并让兼容字段
`preview_indices` 指向实际 display 序列。该澄清只改变 metadata，不改变刺激生成或首批结果。

## 6. 文件结构

```text
configs/
├── paper_parafoveal_clean_hebb.toml
├── paper_parafoveal_clean_nohebb.toml
├── paper_clean_reference_diagnostic.toml
├── paper_recovery_feedforward_baseline.toml
├── paper_recovery_feedforward_calibration.toml
├── paper_memory_carrier_pilot.toml
├── paper_memory_carrier_calibrated.toml
└── paper_shared_decoder_training.toml

predify_experiments/
├── paper_stimuli.py       # 五类映射、balanced targets、no/1/2-back、canvas
├── paper_parafoveal.py    # schema-v4 runner、state carrier、clean reference
├── paper_analysis.py      # grouped SVM、paired bootstrap、recovery/causal CSV
├── recovery_calibration.py# PCoder5 gain sweep 与 held-out ranking
├── paper_shared_decoder.py# 跨图片共同五分类器、配对检验、拟合诊断
└── paper_figures.py       # clean/recovery/carrier/shared-readout 图

Hebbian-CNN/Log/paper_parafoveal_v1/
├── representations/
│   ├── hebb/              # no-prime / one-back / two-back，200 targets each
│   └── nohebb/            # matched 3 files，200 targets each
├── decoding/
│   ├── hebb/
│   │   ├── layer_timestep_decoding.csv
│   │   └── paper_condition_effects.csv
│   └── nohebb/
├── recovery/
│   ├── target_only/       # noise .5 与 occlusion .3，200 targets each
│   ├── feedforward/       # random_init=false matched reference
│   ├── analysis/          # default gain 的 recovery 与 feedforward correction
│   └── calibration/
│       ├── feedforward300/# selection + held-out 共用 reference
│       ├── sweep/         # PCoder5 候选，targets 0–99
│       └── confirm/       # 胜出参数，targets 100–299
├── causal_pilot_calibrated/
│   ├── *.pckl             # 14 个正式 carrier × identity × degradation 文件
│   └── analysis/          # decoding、recovery、error、Same−Different CSV
├── shared_decoder/
│   ├── train/             # 2 个 target-only degradation 文件，200 images each
│   └── analysis/
│       ├── shared_decoder_accuracies.csv
│       ├── shared_decoder_same_vs_different.csv
│       ├── shared_decoder_fit_diagnostics.csv
│       ├── shared_decoders.pckl
│       └── shared_decoder_predictions.pckl
└── figures/
    ├── clean_paper_effects.png
    ├── clean_reference_recovery.png
    ├── feedforward_corrected_recovery.png
    ├── causal_memory_carriers.png
    └── shared_decoder_fc2.png

Hebbian-CNN/Log/paper_smoke/
├── clean_hebb/            # 5-target GPU smoke
├── recovery/              # 5-target clean-reference smoke
├── analysis25/            # 25-target end-to-end decoder smoke
├── causal10/              # 14 个 carrier/identity/degradation smoke files
└── causal10_analysis/      # Same−Different 与 recovery 分析 smoke
```

所有 `Hebbian-CNN/Log` 结果仍由 `.gitignore` 排除；正式数据不会误提交 Git。

## 7. Paper-matched clean benchmark 结果

分析使用 5-fold identity-grouped linear SVM，bootstrap 10,000 次，seed 114514。表中的 CI
是逐 target 配对 bootstrap，尚未对 4 layers × 10 timesteps 做多重比较校正，因此首轮
点位结论应视为机制定位，而不是最终确认。

### 7.1 1-back 的关键早期效应

在 Hebbian 模型中，Conv4 t0：

- 五类解码准确率 no-prime 43.5%，1-back 46.0%；
- 差值 **+2.5 pp，95% CI [+0.5,+5.0]**；
- mean absolute activation 差值 **−0.52%**，95% CI [−1.29,+0.09]。

这与论文的“更低响应、类别信息更明确”方向一致，并且发生在第一个 target predictive
state。但 suppression CI 上界略高于 0，所以没有通过严格联合判据
`decoding lower CI > 0 AND activation upper CI < 0`。

FC2 t1 出现可靠 activation suppression：−0.65%，95% CI [−0.98,−0.34]；其解码差
+2.5 pp，但 CI [−1.5,+6.5] 跨 0。也就是说，两个组成部分出现在不同 layer/time 点，
尚未在预指定的同一深层状态上完整重合。

### 7.2 2-back 与时间演化

2-back 没有呈现论文预期的普遍衰减：FC1 t1 为 +4.0 pp [ +0.5,+8.0]，FC2 t6 为
+2.5 pp [+0.5,+5.0]，FC2 t8 为 +3.5 pp [+1.0,+6.0]。但 FC1 t1 activation 反而
上升，FC2 的 suppression CI 跨 0，仍没有完整联合 phenotype。

1-back 的 Conv4 优势只在 t0 最明确，随后归零；这与旧 foveated 实验中“early head-start
被 recurrence 冲掉”的现象一致。终端 t9 的 one-back 解码差为 Conv4 0、Conv5 −2、
FC1 +0.5、FC2 +2 pp，CI 全部跨 0。

### 7.3 Hebbian 因果控制与终端分类

no-Hebb 下 no/1/2-back 的 logits、隐藏表征和解码结果在数值精度内相同，全部
Prime−No-prime 差为 0。这排除了 target selection、category mapping 和 grouped-SVM
本身制造 1-back 效应的解释。

由于 15% canvas object 很小、而原始 1000-way FC 未针对五类缩放刺激微调，ImageNet
Top-1 处于 0–1.5% floor。Hebbian clean 文件的 t9 Top-1 分别为 no-prime 0.5%、1-back
1.5%、2-back 1.0%；200 targets 下差值 CI 跨 0。该结果不能支持终端分类提升，但也说明
五类 hidden decoding 比原始 1000-way Top-1 更贴近论文的神经表征端点。

## 8. Feedforward-corrected recovery 与 gain calibration

### 8.1 为什么不能用 recurrent t0 作基线

`random_init=true` 时，clean/degraded branches 使用同步随机初始化。两者 t0 表征因共享
随机状态而人为相似，尤其 PCoder5 cosine 接近 0.98；随后网络真正纳入输入信息，二者分离。
因此 t9−t0 为负不代表比 feedforward 更差。正确比较是：

```text
random_init=true recurrent state at each t
minus
matched random_init=false feedforward state at t0
```

按该口径，默认参数在 noise 下 PCoder1–5 全部显著优于 feedforward；occlusion 下
PCoder1–4 显著改善，PCoder5 为 +0.00064 [−0.00098,+0.00219]。

### 8.2 PCoder5 校准

选择集为 targets 0–99，确认集为 targets 100–299；五类平衡且不重叠。12 个候选只改变
无 top-down 输入的 PCoder5，PCoder1–4 固定。选择规则优先最大化 PCoder4/5、两种
degradation 的最差 feedforward-corrected recovery CI，同时要求 FC1/FC2 五类解码不显著
劣于 feedforward。

最优候选为：

```toml
# PCoder5 only
ffm = 0.4
fbm = 0.0
erm = 0.0
```

`ffm=0.4, erm=0` 与 `ffm=0.4, erm=0.01` 几乎相同，说明主要改善来自增加短时 memory
权重，而非 error-gradient gain。held-out 200 targets 上，noise 的 PCoder4/5 cosine 与
normalized L2 全部显著改善；occlusion 的 PCoder4 两指标显著，PCoder5 normalized L2
显著，cosine 为 +0.00087 [−0.00064,+0.00241]。FC1/FC2 没有显著退化。

## 9. 正式 Same/Different × memory-carrier 结果

正式 factorial 使用 targets 100–299、两种 degradation、200 targets/文件：两种
degradation 各一个 no-memory baseline，加上三个有效 carriers × Same/Different，共
14/14 文件。每文件 logits `[200,10,1000]`、recovery `[200,10,5,3]`，五类各 40，均为
有限值。occlusion seeds 为全局 positions 100–299；noise 共享 seed 0 realization。

### 9.1 论文样隐藏表征效应

noise 下，三个 carrier 都在 FC2 的不同早期/中期 timestep 出现 Same−Different 的完整
“解码提高 + 激活降低”：

| carrier | timestep | 五类解码差 | activation 差 |
|---|---:|---:|---:|
| `pcoder_carry` | t2 | **+8.0 pp [ +2.0,+14.5]** | **−1.36% [−1.76,−0.99]** |
| `hebb_and_pcoder` | t3 | **+6.5 pp [+0.5,+12.5]** | **−0.33% [−0.66,−0.02]** |
| `hebb_only` | t4 | **+5.5 pp [+0.5,+10.5]** | **−0.93% [−1.62,−0.25]** |

这些是 4 layers × 10 timesteps 的探索性逐点 CI，尚未做多重比较校正；时间点必须在下一
批独立数据上预指定后才能作为确认性结论。occlusion 有较强解码 head-start，例如联合
carrier 的 FC2 t1 为 +15.5 pp [+9,+22]，但同点 activation suppression CI 跨 0，未通过
严格联合判据。

### 9.2 对象特异 clean recovery

在 t9，Same prime 相对 Different prime 的 PCoder4/5 representation recovery：

| degradation | carrier | PC4 cosine advantage | PC5 cosine advantage |
|---|---|---:|---:|
| noise | `hebb_only` | +0.00153 | +0.00559 |
| noise | `pcoder_carry` | +0.000004 | +0.000238 |
| noise | `hebb_and_pcoder` | +0.00117 | +0.00439 |
| occlusion | `hebb_only` | +0.00134 | +0.00469 |
| occlusion | `pcoder_carry` | +0.000002 | +0.000212 |
| occlusion | `hebb_and_pcoder` | +0.00107 | +0.00384 |

除 occlusion `pcoder_carry` 的 PC4 cosine 下界约等于 0 外，其余均显著为正；normalized
L2 全部同向。Hebbian recovery 约比纯 PCoder carry 大一个数量级并持续到 t9。PCoder carry
则在 t1–t3 较强、随后衰减，这与其早期 FC2 sharpening 但无 t9 优势相吻合。

### 9.3 原始 VGG 终端分类仍未提升

所有 carrier/degradation 的 t9 Same−Different Top-5 都为 0 pp；Top-1 为 0 或 −0.5 pp。
任一 timestep 均没有 Top-1/Top-5 的正向 CI 下界超过 0。绝对 t9 Top-1 仅 0–0.5%，
Top-5 仅 1.0–1.5%。因此本阶段证明了对象特异 recovery 和短暂 hidden sharpening，但没有
证明 torchvision 原始 1000-way FC 的终端性能提升。

## 10. 机制解释与神经科学对应

结果支持两个时间尺度不同的 memory mechanism：

1. **PCoder state carry** 立即把 preview 的生成状态带入 target，t2 最早出现完整
   suppression + decoding enhancement；但状态优势快速衰减，PCoder5 recovery 到 t9 只剩
   约 0.0002；
2. **Hebbian carry** 初期 recovery 可为负，约 t2 后转正，并在 t9 保留更大的对象特异
   clean-reference advantage；它更接近论文讨论的短时突触可塑性；
3. **联合 carrier** 兼有早期生成状态与较持续 Hebbian trace，但并未简单线性叠加，提示
   两类 memory 通过相同 recurrent dynamics 竞争，而不是独立相加。

这种时间演化与 MEG repetition sharpening 的相似处是：Same prime 可在总体响应降低时
提高类别可解码性，而且效应是短暂的。不同处是模型尚未把这一隐藏状态优势转成原始 VGG
决策。原因很具体：15% canvas objects 让 1000-way head 处于严重 floor；该 head 也从未在
paper 五类、PCoder recurrent states 或 parafoveal sequence 上训练。

因此“PCoder 生成能力弱”需要修正为：PCoder 能产生 clean-like prediction，且 Same prime
确实改善深层 clean recovery；当前瓶颈主要是 **readout/task mismatch 与有利时间窗没有被
终端 head 使用**，而不是完全缺乏生成恢复。

## 11. 共享固定五分类读出：已执行结果

### 11.1 设计与数据隔离

两个 degradation 各使用 schedule positions 300–499 的 200 张 target-only 图像训练
`StandardScaler + LinearSVC(C=1)`；每个 layer/timestep 一套分类器。分类器拟合后冻结，
再应用到正式 carrier 实验 positions 100–299 的全部 Same/Different 文件。因此一个比较中
Same 与 Different 使用完全相同的分类权重，准确率差不再来自“为每个条件各自重训一个
最优超平面”。

训练与测试的 ImageNet validation **图像 index 零重叠**。但这是 image-disjoint，不是
synset-disjoint：训练有 167 个 ImageNet labels，测试有 167 个，其中 72 个重叠。固定读出
结果因此能证明跨图片泛化，不能声称对未见 ImageNet object identities 的五大类迁移。
原 identity-grouped cross-validation 仍是更严格的 category-information 指标。

80 个分类器中 78 个在 50,000 iterations 内收敛；两个未收敛模型恰为 noise/occlusion 的
FC2 t0。所有预指定 t2/t3/t4 与终态 t9 模型均收敛。逐模型状态保存在
`shared_decoder_fit_diagnostics.csv`。

### 11.2 预指定早期端点没有复制独立 SVM 优势

| degradation | carrier / endpoint | fixed-head Same−Different |
|---|---|---:|
| noise | `pcoder_carry`, FC2 t2 | −1.0 pp [−4.5,+2.5] |
| noise | `hebb_and_pcoder`, FC2 t3 | 0.0 pp [−4.0,+4.0] |
| noise | `hebb_only`, FC2 t4 | 0.0 pp [−2.5,+2.5] |
| occlusion | `pcoder_carry`, FC2 t2 | +0.5 pp [0.0,+1.5] |
| occlusion | `hebb_and_pcoder`, FC2 t3 | −0.5 pp [−3.5,+2.5] |
| occlusion | `hebb_only`, FC2 t4 | +2.0 pp [−0.5,+4.5] |

这说明“Same 条件中可由独立交叉验证 SVM 提取更多类别信息”不自动等于“同一个固定决策
边界立即提高准确率”。priming 同时移动了特征均值、尺度与方向；条件内最优超平面能够
利用这种变化，而一个在独立 degraded target 上训练的固定超平面未必在相同早期时刻对齐。

### 11.3 唯一终态候选信号

FC2 t9 的六个预览如下：

| degradation | carrier | Same−Different |
|---|---|---:|
| noise | `hebb_only` | +0.5 pp [−1.0,+2.5] |
| noise | `pcoder_carry` | +1.0 pp [0.0,+2.5] |
| noise | `hebb_and_pcoder` | **+2.5 pp [+0.5,+5.0]** |
| occlusion | `hebb_only` | +1.5 pp [−1.0,+4.0] |
| occlusion | `pcoder_carry` | 0.0 pp [0.0,0.0] |
| occlusion | `hebb_and_pcoder` | −1.0 pp [−3.0,+1.0] |

noise 联合 carrier 的绝对准确率为 Same 41.5%、Different 39.0%。配对表中有 5 个
Same-only correct、0 个 Different-only correct；percentile bootstrap 95% CI 排除 0，
符合预定的 CI 判据。但 exact McNemar 双侧 `p=0.0625`，方向性单侧 `p=0.03125`；而且
240 个 layer × timestep × carrier × degradation 单元中有 26 个未校正 bootstrap-positive
点，t9 只有这一点。故它应被称为 **预注册价值很高的方向性候选**，而非已确认的终端
分类提升。

该结果与机制记录相容：PCoder carry 先提供快速但衰减的生成状态，Hebbian trace 提供较慢、
持续的对象特异 clean recovery；只有联合 carrier 在 noise t9 把二者留下的 FC2 位移落到
固定分类面的有利一侧。occlusion 不出现相同结果，提示被遮掉的信息不能仅靠现有 prediction
path 稳定补全，而高斯噪声中的部分证据仍可被 recurrent integration 利用。

原 torchvision 1000-way Top-1/Top-5 结论不变：它仍处于 15% object 的 floor，且所有
carrier 的原始 t9 Top-5 Same−Different 为 0。这里的候选“终端”只指冻结的、任务匹配
五分类 readout，不是原始 VGG ImageNet head。

### 11.4 下一轮确认性实验

下一轮不再搜索 timestep、gain 或 carrier，而只确认：

```text
noise 0.5 + hebb_and_pcoder + FC2 t9 + Same > Different
```

实施时应先在每个 supercategory 内划分 ImageNet synsets，再生成刺激：一组 synsets 只用于
训练 frozen five-category head，另一组只用于正式 target，解决本轮 72-label overlap。
固定当前 PCoder5 `0.4/0/0`、Hebbian 参数、object scale、读出超参数和 seed 规则；使用至少
1000 个新 targets 和多个新的 degradation seeds。主要统计同时报告 paired bootstrap CI、
Same-only/Different-only counts 与双侧 exact McNemar，并预先声明单一主端点，无需再为
4×10 时空搜索校正。还应加入 no-memory/no-prime，使 Same 同时优于 Different 和无记忆基线。

若该确认失败，再升级生成能力：优先让 PCoder4/5 decoder 以 clean lower-layer target 训练，
并在 clean-reference recovery 上选择架构；不要直接以当前 200 targets 的分类差调 decoder。
若目标改为原始 ImageNet Top-1，则另行训练 scale/degradation-compatible 1000-way head，
并明确把它作为 readout adaptation，而非 Hebbian/PCoder 机制本身。

## 12. 复现命令

```powershell
# Paper clean，Hebbian / no-Hebb
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_parafoveal_clean_hebb.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

python -m predify_experiments.paper_parafoveal `
  --config configs/paper_parafoveal_clean_nohebb.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

# Target-only clean-reference diagnostic
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_clean_reference_diagnostic.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

# Matched random_init=false feedforward recovery reference
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_recovery_feedforward_baseline.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

# Calibrated Same/Different × carrier formal run (14 files)
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_memory_carrier_calibrated.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

# Independent target-only data for the shared frozen readout
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_shared_decoder_training.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

python -m predify_experiments.paper_shared_decoder `
  --train-dir Hebbian-CNN/Log/paper_parafoveal_v1/shared_decoder/train `
  --evaluation-dir Hebbian-CNN/Log/paper_parafoveal_v1/causal_pilot_calibrated `
  --output-dir Hebbian-CNN/Log/paper_parafoveal_v1/shared_decoder/analysis `
  --seed 114514 --bootstrap-samples 10000 --max-iter 50000

# Grouped SVM、paired bootstrap、recovery 与 internal-error CSV
python -m predify_experiments.paper_analysis `
  --input-dir Hebbian-CNN/Log/paper_parafoveal_v1/recovery/target_only `
  --output-dir Hebbian-CNN/Log/paper_parafoveal_v1/recovery/analysis `
  --folds 5 --seed 114514 --bootstrap-samples 10000

# Figures
python -m predify_experiments.paper_figures `
  --effects-csv Hebbian-CNN/Log/paper_parafoveal_v1/decoding/hebb/paper_condition_effects.csv `
  --recovery-csv Hebbian-CNN/Log/paper_parafoveal_v1/recovery/analysis/clean_reference_recovery.csv `
  --output-dir Hebbian-CNN/Log/paper_parafoveal_v1/figures `
  --feedforward-comparison-csv Hebbian-CNN/Log/paper_parafoveal_v1/recovery/analysis/recurrent_vs_feedforward_recovery.csv `
  --causal-csv Hebbian-CNN/Log/paper_parafoveal_v1/causal_pilot_calibrated/analysis/causal_same_vs_different.csv `
  --causal-recovery-csv Hebbian-CNN/Log/paper_parafoveal_v1/causal_pilot_calibrated/analysis/causal_recovery_same_vs_different.csv `
  --shared-decoder-effects-csv Hebbian-CNN/Log/paper_parafoveal_v1/shared_decoder/analysis/shared_decoder_same_vs_different.csv
```

正式验证使用 `predify2021` Conda 环境。当前全量单元测试为 **54/54 通过**；另有 clean、
recovery、25-target decoder、14-condition carrier、正式 14-file carrier 与 2-file shared
readout training GPU runs。正式审计确认：14/14 文件可反序列化，logits 均为
`[200,10,1000]`，recovery 均为 `[200,10,5,3]`，五类各 40，全部有限；共享训练/测试
图像 index overlap 为 0。安装的 scikit-learn 为 1.5.2，environment 约束为 1.5。
