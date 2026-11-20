# 对标 Wynn、Bezsudnova & Jensen：Parafoveal Priming 与 Neural Sharpening

> **执行状态（2026-09-02）**：本文档中的 paper-matched clean benchmark、
> clean-reference diagnostic 与 memory-carrier runner 已实现。200-target clean/no-Hebb
> 与两种 200-target recovery 条件已经完成。经 matched feedforward baseline 修正及
> PCoder5 校准后，14-file carrier factorial 与共享固定五分类读出均已完成；早期基于
> recurrent t0 的 NO-GO 判断不再使用。原始 1000-way head 仍无提升；固定五分类读出在
> noise/联合 carrier/FC2 t9 出现 +2.5 pp 方向性候选，但 exact McNemar 双侧 p=0.0625，
> 且训练/测试 synsets 未完全分离，因此仍需确认。完整
> schema、文件树、数值结果和决定见
> [`paper_parafoveal_v1_execution.md`](paper_parafoveal_v1_execution.md)。

## 1. 对标对象与结论口径

主要对标论文为 Wynn、Bezsudnova 与 Jensen 的预印本
[*Neural Sharpening of Object Categories Through Parafoveal Priming*](https://www.biorxiv.org/content/10.1101/2025.05.16.654463v1.abstract)
（bioRxiv v1，2025）。严格说，该研究是全头 MEG，而不是单神经元或颅内电生理；
它仍然是本项目最直接的时间分辨神经生理对标。

论文中的“classification improvement”指的是：从 MEG 多传感器模式中，用 SVM
更准确地解码 foveal target 的五个物体大类。它不是被试在 target 当下完成图像分类，
也不是一个预训练视觉分类器的 Top-1。因此，本项目需要区分三个 endpoint：

1. **神经表型主指标**：隐藏层表征的类别可解码性是否提高；
2. **神经效率指标**：总体 activation 是否同时降低；
3. **工程外推指标**：ImageNet FC Top-1/Top-5 是否提高。

前两项共同对应论文的 neural sharpening；第三项是更强、但并非论文结论所必需的
下游鲁棒性检验。

## 2. 论文的实验范式

### 2.1 刺激与时序

每个 viewing trial 的顺序是：

```text
fixation 100 ms
  → parafoveal display 1：左右各一个物体，150 ms
  → fixation 100 ms
  → parafoveal display 2：左右各一个物体，150 ms
  → fixation 100 ms
  → foveal target：中央单物体，500 ms
```

三个条件只改变 target 的完全相同图像是否在前两个 parafoveal displays 中出现：

| 条件 | target 的前次出现 | prime onset 到 target onset | 解释 |
|---|---|---:|---|
| no prime | 两个 display 都未出现 | — | 基线 |
| 1-back prime | display 2 | 250 ms | 直接 preview |
| 2-back prime | display 1 | 500 ms | 中间隔一个双物体 display |

prime 与 target 是**完全相同的图片**，而不只是相同类别。prime 左右位置平衡。
各图片的物理尺寸相同；论文没有人为降低 parafoveal prime 的对比度或加入 blur，
其低视敏度来自视网膜偏心位置。no-prime 和 prime 条件中的其他 parafoveal 图片取自
target 以外的物体大类。

刺激来自五个大类：animals、clothing、food、plants、vehicles。论文共使用大量不同
object identity 和 image exemplar，使类别解码不是对少数重复图片的记忆。

### 2.2 论文的关键测量与结果

| 证据层 | 操作指标 | 主要结果 |
|---|---|---|
| 行为 | viewing phase 后的 old/new recognition，d′ | 1-back 0.66，高于 no-prime 0.58；2-back 0.57，与 no-prime 无差异 |
| response magnitude | foveal target onset 后 ERF | 1-back 在约 150–210 ms 出现 repetition suppression；2-back 效应较弱 |
| category specificity | MEG sensor pattern 的五类 SVM decoding | 约 100 ms 后可解码；在预设 150–210 ms 窗内，1-back 约 0.55，高于 no-prime 约 0.54；2-back无可靠早期增益 |
| late dynamics | ERF 与 decoding timeline | 1-back 约 305–360 ms 有晚期 response enhancement；解码还出现更晚的显著区间 |
| oscillation | 40–80 Hz induced gamma | 没有得到稳健的 priming modulation |

因此论文最有辨识力的表型不是简单的“重复后响应更低”，而是：

```text
1-back：target response ↓  且  target category decodability ↑
2-back：两种效应减弱或在关键早期窗口消失
```

这种时间距离依赖性使结果难以被单纯的静态缩放解释。论文讨论提出的候选机制正是：
同一对象表征单元之间的 recurrent connections、短时 Hebbian strengthening，以及
lateral inhibition 对弱响应单元的抑制。

## 3. 当前模型与论文的逐项对应

| 论文要素 | 当前实现 | 对齐程度与影响 |
|---|---|---|
| 两个连续 parafoveal displays | 一个 even prime trial | **未对齐**；无法构造真正 2-back 和 intervening-display interference |
| 每个 display 左右两个物体 | `parafoveal_half_field` 左右拼接 | 部分对齐 |
| prime/target 完全相同图片 | Prime even 与随后 odd target 相同 | 对齐 |
| no-prime 中 target 类别不出现在 preview | NonPrime 用 schedule 中另一图片 | 未严格约束大类，存在 category-level priming 混淆 |
| prime/target 同一物理大小，偏心位置不同 | half-field 另加 contrast 0.5、blur、downsample | **过度退化且不是论文操作** |
| clean natural images | 已有 clean，也扩展到 noise/occlusion | clean 才是直接复制；退化条件属于新外推 |
| 1-back、2-back、no-prime | Prime、NonPrime | 只近似 1-back/no-prime；缺 2-back |
| 连续脑状态与短时突触变化 | Hebbian weight 跨 even→odd；PCoder state 每 trial reset | Hebbian carry 对齐候选机制；持续 recurrent state 未对齐 |
| ERF response magnitude | Conv/FC activation magnitude | 可作功能类比，但不等于 MEG 场强 |
| 五类 MEG MVPA | 200 类内几何、ImageNet 1000-way FC Top-1 | **主要 endpoint 未直接对齐** |
| delayed recognition d′ | 无后续 episodic-memory task | 未对齐；ImageNet Top-1 不能替代 |
| 150–210 ms 时间窗 | recurrent t0–t9 | 没有经过标定，不能把某个 timestep 直接称为 175 ms |

### 3.1 对现有结果的重新解释

现有 foveated/half-field 结果可以回答模型工程问题，但不能直接作为对论文的成功或失败
复制：

- Conv4/Conv5 的 activation 下降与 scale-invariant Fisher/类内集中度改善，是与论文
  相容的辅助证据；
- noise 0.5 下 Conv4/5 sharpening 与 ImageNet Top-1 下降并存，并不反驳论文，因为
  论文的 sharpening 是 MEG category decoding，不是既有任务 head 的准确率；
- foveated full-image prime 不是论文的关键条件。论文的关键恰恰是“先 parafoveal、
  后 foveal”的跨位置重复；
- no-Hebb Prime/NonPrime target logits 完全相同，只说明现有实现中 prime 的作用由
  Hebbian weight 承载，不说明神经模式解码没有变化；后者必须直接训练 readout 测量。

此外，重新检查现有 target logits timeline 后发现，t9 汇总掩盖了一个瞬时早期效应：

| layout / target | t0 P−NP Top-1 pp，95% CI | t1 | t9 |
|---|---:|---:|---:|
| foveated / clean | +3.8 [+2.1,+5.6] | −0.4 [−1.3,+0.6] | −0.8 [−1.7,+0.1] |
| foveated / noise 0.5 | +4.6 [+2.9,+6.3] | −0.7 [−1.9,+0.5] | −1.6 [−2.6,−0.6] |
| foveated / occlusion 0.3 | +6.5 [+4.6,+8.5] | +0.4 [−0.7,+1.5] | 0.0 [−0.9,+0.9] |
| half-field / clean | +0.9 [−0.5,+2.3] | −0.4 [−1.2,+0.4] | −0.1 [−0.7,+0.5] |
| half-field / noise 0.5 | +0.8 [−0.4,+2.0] | −0.4 [−1.2,+0.4] | −0.2 [−0.8,+0.3] |
| half-field / occlusion 0.3 | +2.4 [+0.8,+3.9] | 0.0 [−0.7,+0.7] | −0.2 [−0.8,+0.4] |

置信区间为 1000 个相同 targets 的 10,000 次配对 bootstrap，seed 114514。该结果是
新增的探索性 timeline 分析，不是预注册检验。它提示 Hebbian prime 可能提供一个很短的
分类 head-start，随后被 target 内 recurrence 消除，noise 0.5 的 foveated 条件甚至
从早期正效应翻转为晚期负效应。

对应数值记录在
`Hebbian-CNN/Log/comparisons_v1/priming_timeline_selected.csv`。

这与论文“效应具有明确时间窗”的结构相似，但不能直接称为论文复现：当前 t0 已包含
`random_init=true` 下的第一次 predictive update；1000-way torchvision FC Top-1 也
不是五类隐藏表征 MVPA。它提出了一个明确机制问题：为什么 prime 带来的初始 bias 在
t1 被冲掉，以及后续 recurrent/error update 是否对 prime-conditioned state 过度修正。

## 4. 优先实施的 paper-matched 实验

### 4.1 Phase P0：先复制范式，不加入噪声或遮挡

新增 `paper_parafoveal_sequence` trial builder：

1. 每个 sample 生成 preview-1、preview-2、foveal-target 三个连续刺激；
2. 每个 preview 均含左右两个物体；target 为中央单物体；
3. 构造 no-prime、1-back、2-back 三条件；
4. prime 与 target 使用完全相同图片，左右位置平衡；
5. 所有其他 preview 图片来自 target 之外的大类；
6. 不额外降对比、blur 或 downsample；首先复制论文的 screen-level manipulation；
7. 每个 preview 后都允许网络完成固定步数并按预先定义规则更新短时 memory；每个完整
   trial 开始前 reset，target 前不清除该 trial 内应保留的 Hebbian state。

数据优先使用与论文一致的五个 superordinate categories。可选实现顺序为：

- 首选 THINGS-compatible stimulus subset；
- 若数据暂不可得，则先建立 ImageNet→五大类的人工审校映射，每类包含多个 object
  identities，且按 identity 分割 train/test，防止 exemplar leakage。

### 4.2 Phase P1：建立 MEG-MVPA 的模型对应物

对每个 target 的每层、每 timestep 保存 representation，并做五类 time-resolved
linear SVM decoding：

- 主 readout：Conv4、Conv5 的 global-pooled vector；
- 补充 readout：FC1、FC2；
- 同一 object identity 的不同图片不能跨 train/test 泄漏；
- 三条件使用完全一致的 folds 和样本平衡；
- 主对比：`decoding(1-back) − decoding(no-prime)`；
- temporal-specificity 对比：
  `[1-back − no-prime] − [2-back − no-prime]`；
- bootstrap/permutation 在 sample 或 object-identity 层配对完成。

同时报告 activation magnitude。paper-matched 成功标准为：至少一个预先指定的深层
visual readout 区域出现 1-back activation↓ 与 category decoding↑，且 2-back 效应更弱。
normalized within-class variance 和 Fisher separability保留为机制辅助指标，不替代
MVPA 主终点。

作为工程性次级指标，再报告逐 timestep ImageNet Top-1、true-class margin、达到固定
decision threshold 的最早 timestep，以及 margin/accuracy 的时间积分。现有 t0 正效应
说明“更早达到可用证据”可能比“提高 t9 渐近准确率”更接近 parafoveal head-start。

网络 timestep 与毫秒之间暂不做一一映射。先报告完整 `layer × timestep` effect map；
如果以后要声称复现 150–210 ms，应使用独立时间标定规则，而不是从显著结果反推。

### 4.3 Phase P2：机制消融

在 P0/P1 固定的 stimulus schedule 上比较：

| 条件 | Hebbian carry | PCoder state carry | lateral inhibition | 回答问题 |
|---|---:|---:|---:|---|
| no-memory | × | × | 固定 | 纯视觉序列基线 |
| Hebb only | ✓ | × | ✓ | 当前短时突触机制是否足够 |
| recurrence carry only | × | ✓ | ✓ | 持续预测状态是否必要 |
| Hebb + recurrence | ✓ | ✓ | ✓ | 两种记忆是否协同 |
| no inhibition | ✓ | 按最佳条件 | × | sharpening 是否依赖弱单元抑制 |

该阶段先以论文表型为准，不以寻找最高 Top-1 为准。这样可以直接检验论文作者提出的
“recurrent + short-term Hebbian + lateral inhibition”解释，而不是只证明某组超参数
能提高分类。

## 5. 噪声、遮挡与终端性能：作为第二层外推

clean-reference diagnostic 与 memory-carrier causal pilot 不应被视为和 paper-matched
实验竞争的另一套计划。三者依次回答不同问题：

```text
paper-matched clean benchmark
  │  模型能否产生论文的 suppression + category-decoding sharpening？
  ▼
clean-reference target-only diagnostic
  │  PCoder 单独面对退化输入时，是否真的向 clean representation 恢复？
  ▼
Same/Different clean-prime causal pilot
  │  哪一种跨 preview→target memory carrier 能传递对象特异信息？
  ▼
held-out terminal confirmation
     恢复与神经表型收益能否进一步转成 margin / Top-1 收益？
```

### 5.1 共用基础设施先行

在第一次 paper-matched run 前同时完成两套共用记录能力：

1. 精确的 no-prime/1-back/2-back 双 preview sequence builder；
2. clean-reference forward cache，以及每样本、每层、每 timestep 的 recovery metrics。

clean-reference 建议固定为同一网络、同一图片在 clean input 下的 reference，而不是随
degraded recurrence 一起移动的内部 target。至少记录：

- `representation_degraded(t) ↔ representation_clean(t)` 的 cosine、normalized L2；
- `prediction_degraded(t) ↔ clean lower-layer reference` 的 cosine、MSE；
- 内部 `prediction ↔ degraded lower-layer target` error，作为模型自洽性对照；
- true-class margin、correctness transition 和 activation。

正式文件保存标量距离即可；只为固定的小型 diagnostic subset 保存完整 prediction/
representation tensors，避免 PCoder1–5 全张量使文件规模失控。

### 5.2 Phase G：target-only clean-reference diagnostic

在 paper-matched clean benchmark 之后、memory factorial 之前，先关闭 Hebbian carry 和
跨 trial PCoder carry，仅输入 degraded target。首轮使用 noise 0.5、occlusion 0.3，
200 targets 即可；若方向明确，再扩展 noise 0.75、occlusion 0.4 与完整 1000 targets。

该阶段的 go/no-go 判据是：相对 target 的首个 predictive state，至少一个与分类有关的
深层 representation 到 clean reference 的距离随 recurrence 可靠下降。若内部
degraded-target error 下降而 clean-reference distance 不下降，说明模型只是在提高受损
层之间的自洽性，不是在恢复正确图像信息。若 clean-reference distance 完全不改善，
直接测试 PCoder carry 的成功概率较低，应先检查 decoder objective、容量和 recurrent
gain。

### 5.3 Phase C：200-target memory-carrier causal pilot

通过 Phase G 后，使用论文效应最强的 **1-back** schedule：preview-1 为 distractors，
preview-2 含 clean prime，随后呈现 degraded foveal target。2-back 已在 clean
paper-matched benchmark 中检验，不在首轮 degraded factorial 中重复展开。

首轮 factorial：

| memory carrier | Same clean prime | Different clean prime | 主要判据 |
|---|---:|---:|---|
| no memory | 运行一次并验证两者应相同 | 可复用 | 实现负控制 |
| Hebb only，PCoder reset | ✓ | ✓ | 论文提出的短时突触 + inhibition 是否传递对象信息 |
| PCoder carry，no-Hebb | ✓ | ✓ | 持续 predictive/generative state 是否传递对象信息 |
| Hebb + PCoder carry | ✓ | ✓ | 两种 memory 是否协同或相互干扰 |

`Different` 首轮遵循论文 no-prime 逻辑，使用 target 之外大类、视觉难度匹配的 clean
图片。若 Same 优于 Different，再增加“同大类、不同 exemplar”以区分 exact-image 与
category-level priming，不在首轮同时加入以避免条件爆炸。

每个 cell 同时使用四层 endpoint：

1. **paper phenotype**：activation suppression 与五类 hidden-state decoding；
2. **recovery**：到 clean-reference prediction/representation 的距离；
3. **decision dynamics**：逐 timestep margin、首次越过 decision threshold、t0→t9
   right/wrong transitions；
4. **terminal engineering**：t9 Top-1/Top-5。

因果证据顺序为：Same 相对 Different 先改善 clean-reference recovery，再改善隐藏类别
解码，最后才讨论 Top-1。若只有 Top-1 改善而 recovery/decoding 均不改善，应优先视为
偶然 calibration，而不是生成式 priming。

### 5.4 Phase H：held-out confirmation

仅选择 causal pilot 中预先胜出的一个 memory carrier，在未用于选择 timestep、gain 或
carrier 的 held-out targets 与多个 degradation seeds 上确认：

- Same−Different recovery 改善；
- Same−Different hidden decoding 改善；
- 若声称终端收益，再要求 Same−Different 与 Same−NoPrime 的 Top-1 置信区间均不跨 0；
- clean target 不应出现不可接受的准确率损失。

只有完成这些门槛后，再引入更强 decoder 或更大参数搜索。

### 5.5 与原退化外推假设的关系

在 degraded pilot 中采用：

```text
clean parafoveal same-image 1-back prime
  → degraded foveal target
```

这里 prime 应保持 clean，噪声/遮挡只施加到 target。这样 prime 才提供额外正确信息，
避免现有“prime 与 target 共享同一 corruption realization”强化污染特征的问题。

建议先做 noise 0.5 与 occlusion 0.3，小样本筛查顺序如下：

1. hidden-category decoding 是否因 same prime 相对 different/no prime 提高；
2. recurrent representation 到 clean-reference representation 的距离是否下降；
3. ImageNet true-class margin、Top-1 是否随后改善；
4. 若前两项均不改善，再评估 decoder 的训练目标与容量；若前两项改善而 Top-1 不改善，
   瓶颈更可能在 classifier alignment 或深层信息传播，而不是“生成能力不足”。

因此，寻找正向 terminal priming 仍值得继续，但它应被表述为对论文机制的鲁棒性外推，
而不是论文复制的成败标准。

## 6. 结果记录要求

paper-matched schema 应在现有 v3 字段之外增加：

- `paper_reference_doi`、`sequence_layout`；
- `condition = no_prime | one_back | two_back`；
- preview-1/preview-2/target 的 dataset index、object identity、superordinate category；
- 每个 preview 的 target identity 是否出现及左右位置；
- 三个刺激各自的 recurrent timestep 数；
- Hebbian update/commit 次序、pair 内 PCoder state 是否保留；
- target 的逐层逐 timestep activation、representation 和 logits；
- 五类 decoder 的 fold/identity split、seed、逐样本 correctness/decision score；
- 所有 paired effect、置信区间与多重比较规则。

输出目录应与现有工程扩展分离，例如：

```text
Hebbian-CNN/Log/paper_parafoveal_v1/
├── representations/
│   ├── hebb/
│   ├── nohebb/
│   └── ablations/
├── decoding/
│   ├── layer_timestep_decoding.csv
│   ├── activation_suppression.csv
│   ├── one_vs_two_back_effect.csv
│   └── bootstrap_samples.npz
├── figures/
└── REPORT.md
```

原始逐 trial 结果继续放在 `.gitignore` 排除的 `Hebbian-CNN/Log`；实验协议、配置模板、
分析代码与不含大数组的汇总表进入版本控制。

## 7. 当前项目的结论边界

现阶段可以说：模型中已经出现局部 repetition suppression 与部分 scale-invariant
representation concentration；Predify recurrence 也能改善中等 Gaussian noise 的
ImageNet 分类。但尚不能说已经复现 Wynn 等人的主要结果，因为还没有完成精确的
no/1-back/2-back 序列，也没有用匹配五大类的隐藏表征 MVPA 作为主指标。

反过来，现有 terminal Top-1 的 null/negative priming 也不能否定论文的 neural
sharpening。下一步最有效的信息增益来自先完成 paper-matched clean benchmark，再决定
是否需要加强 PCoder 的生成/恢复能力。
