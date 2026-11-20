# Predify–Hebbian Priming：阶段性结果、机制解释与下一步实验

> **Paper-matched 阶段更新（2026-09-02）**：Stage P、Stage A、PCoder5 calibration、
> 14-file carrier factorial 与 shared frozen readout 均已正式完成。
> clean 1-back 在 Conv4 target t0 出现 Hebbian-dependent 五类 decoding +2.5 pp
> [95% CI +0.5,+5.0]，activation 同向下降但 CI 略跨 0；no-Hebb 效应为 0。
> 后续发现 synchronized `random_init=true` 使 recurrent t0 成为偏高且不公平的 recovery
> 基线。改用 matched `random_init=false` feedforward 后，默认 recurrence 在 9/10 个
> PCoder×degradation cells 显著改善；PCoder5 校准选择 `ffm=0.4, fbm=0, erm=0`。
> 正式 carrier 结果显示早期 suppression + decoding enhancement 与持续对象特异 recovery。
> 原始 1000-way head 仍无终端提升；共享固定五分类器在 noise/联合 carrier/FC2 t9 有
> +2.5 pp 候选，但 exact McNemar 双侧 p=0.0625，需 synset-disjoint 新样本确认。
> 详见
> [`paper_parafoveal_v1_execution.md`](paper_parafoveal_v1_execution.md)。

## 1. 一句话结论

到目前为止，证据支持两个彼此独立的结论：

1. **Predify recurrence 本身确实能改善中等 Gaussian noise 下的终端分类**，但对
   clean、遮挡和极强噪声没有显著净收益；
2. **原始 torchvision 1000-way head 的跨 trial priming 没有提高终端分类**。Foveated full prime 不仅没有
   优于 parafovea，在 noise 0.5 下还产生显著负效应，尽管 Conv4/Conv5 同时出现
   局部 repetition suppression + scale-invariant sharpening；后续任务匹配的固定五分类
   readout 出现一个方向性终态候选，但尚未达到确认性证据标准。

这说明“recurrent inference 有用”“局部表征 sharpening 存在”和“prime 改善最终
决策”是三个不同命题，不能互相替代。

结合主要对标论文 Wynn、Bezsudnova & Jensen (2025) 后，还必须增加一个口径修正：
该论文以 **MEG 隐藏神经模式的类别解码提高**定义 sharpening，而不是以 foveal
target 的即时分类行为或预训练网络 Top-1 定义。因此当前 Top-1 null/negative 是重要
工程结果，但尚不是对论文 neural-sharpening 假说的直接否定。完整逐项对标见
[`biorxiv_654463_alignment.md`](biorxiv_654463_alignment.md)。

## 2. 实验空间与因果对比

当前结果横跨七个操作维度：

| 维度 | 水平 |
|---|---|
| 网络 | torchvision vanilla VGG16；Predify VGG16 |
| recurrent time | vanilla 单次 forward；Predify t0–t9 |
| Hebbian state | active Hebbian；no-Hebb pass-through |
| repetition | Prime；NonPrime |
| prime layout | parafoveal half-field；foveated full-image |
| degradation | clean；noise 0.25–2.0；occlusion 0.1–0.5 |
| observation level | Top-1/Top-5；logit margin；PCoder error；Conv4/5、FC1/2 geometry |

为了避免在多维结果中混淆机制，使用以下四个因果对比：

```text
A. Predify system effect
   no-Hebb Predify(t) − vanilla VGG16

B. Generic Hebbian effect under an unrelated prime
   Hebb NonPrime − no-Hebb NonPrime

C. Repetition-specific Hebbian effect
   (Prime − NonPrime)_Hebb − (Prime − NonPrime)_noHebb

D. Layout moderation
   repetition effect_foveated − repetition effect_parafovea
```

no-Hebb Prime/NonPrime 的 odd logits 逐元素完全相同，因此当前 C 在数值上等同于
active Hebbian 内的 Prime−NonPrime，但保留差分中的差分写法仍有助于防止未来实现
改变时误归因。

## 3. 结果层次一：Predify recurrence 是否改善分类

### 3.1 t9 相对 vanilla

| target condition | vanilla Top-1 | Predify t9 Top-1 | 差值 pp，配对 95% CI | 结论 |
|---|---:|---:|---:|---|
| clean | 72.2% | 72.1% | −0.1 [−0.8,+0.6] | null |
| noise 0.25 | 58.8% | 60.4% | +1.6 [+0.6,+2.7] | 显著改善 |
| noise 0.5 | 35.3% | 38.7% | +3.4 [+1.9,+5.0] | 显著改善 |
| noise 0.75 | 14.8% | 18.1% | +3.3 [+2.0,+4.7] | 显著改善 |
| noise 1.0 | 5.5% | 6.8% | +1.3 [+0.4,+2.3] | 显著改善，但接近低性能区 |
| noise 2.0 | 0.7% | 0.7% | 0.0 [−0.3,+0.3] | floor |
| occlusion 0.1 | 69.9% | 70.5% | +0.6 [−0.1,+1.4] | null |
| occlusion 0.2 | 67.4% | 67.3% | −0.1 [−1.0,+0.8] | null |
| occlusion 0.3 | 60.6% | 60.5% | −0.1 [−1.0,+0.7] | null |
| occlusion 0.4 | 47.1% | 47.0% | −0.1 [−1.2,+0.9] | null |
| occlusion 0.5 | 38.5% | 38.7% | +0.2 [−1.1,+1.5] | null |

Predify 的净收益呈倒 U 型：中等 noise 最清楚，clean 没有余量，noise 2.0 已进入
信息 floor；遮挡虽然在个别 timestep 有恢复，但 t9 不显著超过 vanilla。

### 3.2 recurrent refinement

- 当前 `random_init=true` 的 t0 不是 feedforward，而是第一个 predictive state；
  它在所有非地板条件都远低于 vanilla。
- t0→t1 是主要恢复阶段；中等 noise 的额外改善延续到 t4–t6。
- t1→t9 Top-1 增益：noise 0.25 为 +2.4 pp、0.5 为 +5.5 pp、0.75 为
  +3.7 pp、1.0 为 +1.7 pp，配对 CI 均为正。
- noise 0.5 中，t1→t9 有 6.9% targets 从错误变正确，1.4% 从正确变错误，净增
  5.5 pp。这是 selective rescue，并不是所有样本的 logits 都等比例改善。
- 高噪声下平均 true-class margin 反而进一步变负。例如 noise 0.75 从 −2.10
  变为 −2.72，noise 1.0 从 −3.01 变为 −4.16。这意味着 recurrence 可以救回一批
  decision-boundary 附近样本，同时让另一部分困难样本更自信地错误。

## 4. 结果层次二：Foveated priming 是否改善分类

三个代表点的完整路径如下：

| 条件 | vanilla | no-Hebb Predify | Hebb NonPrime | Hebb Prime | Foveated P−NP |
|---|---:|---:|---:|---:|---:|
| clean | 72.2% | 72.1% | 71.8% | 71.0% | −0.8 pp |
| noise 0.5 | 35.3% | 38.7% | 38.0% | 36.4% | −1.6 pp |
| occlusion 0.3 | 60.6% | 60.5% | 60.8% | 60.8% | 0.0 pp |

配对差分中的差分：

- clean：−0.8 pp，95% CI [−1.7,+0.1]；
- noise 0.5：−1.6 pp，[−2.7,−0.6]，显著负；
- occlusion 0.3：0.0 pp，[−0.9,+0.9]。

与 parafovea 相比，noise 0.5 的 foveated layout 进一步降低 effect 1.4 pp，
95% CI [−2.4,−0.4]；clean 与 occlusion 0.3 的布局差异均为 null。

因此，“更中心、更完整、更清晰的空间布局”本身不是当前缺失的环节。

但上述结论只针对 t9。按同一 1000 targets 重新检查 timeline 后，foveated Prime−NonPrime
Top-1 在 t0 分别为 clean +3.8 pp [95% CI +2.1,+5.6]、noise 0.5 +4.6 pp
[+2.9,+6.3]、occlusion 0.3 +6.5 pp [+4.6,+8.5]，到 t1 即消失或转负。half-field
仅 occlusion 0.3 在 t0 可靠为正（+2.4 pp，[+0.8,+3.9]）。这提示现有 priming 更像
短暂 processing head-start，而不是渐近 decision improvement；但 t0 仍是 predictive
state，且 FC Top-1 不是 Wynn 等人使用的五类 MEG-MVPA，故属于探索性辅助证据。

## 5. 结果层次三：表征 sharpening 与终端分类为什么分离

严格 complete sharpening 定义为：

```text
activation ↓
normalized within-class variance ↓
normalized Fisher separability ↑
```

Foveated 结果：

| 条件 | FC2 | FC1 | Conv5 | Conv4 |
|---|---|---|---|---|
| clean | suppression only | concentration without suppression | complete | variance +0.004%，近失 |
| noise 0.5 | suppression only | concentration without suppression | complete | complete |
| occlusion 0.3 | suppression only | concentration without suppression | complete | variance +0.006%，近失 |

最关键的反例是 noise 0.5：Conv4、Conv5 都满足严格 sharpening，但 Top-1 priming
显著下降 1.6 pp。局部 feature geometry 更集中，并不保证：

- 类别决策方向保持正确；
- classifier margin 增加；
- 被压缩的是 nuisance dimensions，而不是有用弱特征；
- sharpening 能继续传播到 FC1/FC2。

当前 FC 参数 `coeff=0`，FC1/FC2 的变化是 upstream 传播结果，不是 FC 自身的
Hebbian 学习。FC1 activation 上升而 normalized variance 下降；FC2 activation
下降但 variance 上升，都不满足完整全网络 sharpening。

## 6. 当前计算机制

### 6.1 Predify target 内 recurrence

每个 PCoder 的状态更新近似为：

```text
representation(t+1)
  = 0.8 × feedforward
  + 0.1 × feedback
  + 0.1 × memory
  − 0.01 × prediction-error gradient
```

高层 representation 经由 decoder 预测下一低层 activity；prediction error 对高层
representation 的梯度参与下一步修正。五组反馈 decoder 在 clean natural-image
features 上以 reconstruction objective 训练，正式推理时冻结。该设计与 Predify
论文一致：用 recurrent feedback 把受损 representation 推向 clean manifold。

但当前权重明显 feedforward-dominant：每步 80% 都重新注入受损 bottom-up drive，
feedback 只有 10%，error-gradient 只有 1%。PCoder5 还没有更高层 feedback，因此
系统顶部不存在独立的 semantic prior。

### 6.2 当前跨 trial Hebbian memory

当前 prime 与 target 之间真正保留的只有 Conv4/Conv5 channel association：

```text
spatial feature map
  → global average pool（丢弃空间位置）
  → one-shot coactivation matrix
  → repeated target channel boost
  → global channel inhibition
  → rescale the full feature map
```

even prime 的最终 recurrent state 生成一次 Hebbian update，odd target 使用该
weight，随后在下一对 trials 前清空。FC Hebbian coefficient 为零。

这个 memory 能做的是“哪些深层 channels 应一起增强/抑制”，不是保存像素级或局部
空间 template。它无法直接补全被遮挡的位置，也无法区分 object feature 与恰好重复的
noise pattern。

### 6.3 一个容易被忽视的关键点：PCoder state 没有跨 trial 保留

`run_recurrent_sequence` 在每个 even prime 和 odd target 开始时都调用
`model.reset()`。因此：

- clean/degraded prime 建立的 PCoder representations、predictions 和 gradients
  在 target 前全部清空；
- target PCoder 只对当前 degraded target 做 recurrent inference；
- prime 只能通过 Hebbian channel weights 间接改变 target inference。

所以当前实验并没有真正测试“prime 形成生成式先验，随后 PCoder 用该先验恢复 target”。
它测试的是“prime 形成一次 channel-gain association，是否有利于下一个 target”。

## 7. 对“PCoder 生成能力较弱”的判断

这个判断有一部分成立，但需要拆成三层。

### 7.1 已有证据支持的部分

1. **强 noise 与 occlusion 没有获得相对 vanilla 的终端净收益。**
2. **PCoder4/5 保存的内部 prediction error 没有随 t1→t9 下降。** 所有 11 个点中，
   PCoder4 平均 error 上升约 28–40%，PCoder5 上升约 50–67%。由于各层 target 本身
   也随 recurrence 改变，这不能简单解释为单一步长“沿错梯度”，但说明耦合系统
   并未表现为一个单调下降的深层 reconstruction objective。
3. **Foveated exact prime 改善了 PCoder5 error，却没有改善分类。** 相对 NonPrime，
   PCoder5 t9 error 下降约 3.4–5.5%，PCoder4 error 上升约 0.4–0.9%；noise 0.5
   分类同时下降。这更像层间 error redistribution，而不是全局信息恢复。
4. decoder 每层主要是单个 transpose convolution（加可选 ReLU）；这里的
   “generative”是 feature reconstruction，不是拥有强语义 latent 的现代图像生成器。

### 7.2 目前不能由保存结果回答的部分

保存的 prediction error 是“对当前受损 lower-layer target 的内部一致性”，不是
“相对 clean image/clean feature 的恢复误差”。现有结果没有保存：

- PCoder1 prediction 与 clean normalized image 的距离；
- 每层 recurrent representation 与 clean reference representation 的距离；
- 补全区域与真实被遮挡内容的误差；
- prime state 是否把 target 拉向正确 class manifold。

因此不能仅凭内部 MSE 断言 decoder 没有生成能力。必须增加 clean-reference
reconstruction assay。

### 7.3 更根本的问题

即使 decoder 很强，当前 priming pipeline 也在 target 前 reset PCoder state，而且
prime 与 target 使用相同 degradation policy。noise 条件还共享同一 noise
realization，exact repetition 可能把 corruption-specific channel pattern 一起写入
Hebbian memory。当前 negative priming 未必首先来自 decoder 容量不足；它也可能来自：

- 没有跨 trial semantic/generative state；
- prime 本身不比 target 提供更多信息；
- one-shot channel boost 过度强化了 noise-correlated features；
- global inhibition 抑制了分类仍需的弱 features。

## 8. 神经科学对应与边界

经典层级 predictive coding 提出，高层反馈预测低层 activity，feedforward 通路传递
残差 error；Predify 保留了“层级反馈 + prediction error 修正”这一计算抽象。参见
[Rao & Ballard (1999)](https://www.nature.com/articles/nn0199_79) 和
[Predify (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/75c58d36157505a600e0695ed0b3a22d-Paper.pdf)。

人类 V1 研究曾观察到 expectation 降低 response amplitude、提高 multivariate
decodability，并且信息改善与行为改善相关；这与“更低激活、更精确表征”的目标最
接近。[Kok, Jehee & de Lange (2012)](https://jeheelab.org/kokjeheelange2012.pdf)

灵长类 IT/V2 中确实存在 repetition suppression，但 suppression 可以在不同区域
独立出现，也不自动等价于 behavioral priming；单神经元工作也专门区分了简单 scaling
与 sharpening。[McMahon & Olson (2007)](https://journals.physiology.org/doi/abs/10.1152/jn.01042.2006)、
[Williams & Olson (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9678433/)

当前模型与脑机制的相似性是功能层面的，而非逐部件同构：

- 相似：层级 feedback、error-related correction、短时 repetition-dependent gain、
  suppression 与 population sharpening 可以共存；
- 不同：显式 backprop gradient、全局平均池化、all-to-all channel matrix、人工阈值、
  trial 边界 reset，以及缺少持续 semantic state；
- 因而，观察到 activation↓ + normalized Fisher↑ 只能说出现了类似的计算表型，
  不能证明模型实现了相同的神经机制。

## 9. 是否继续寻找正向 priming？

**建议继续，但先把“复现论文神经表型”和“提高 ImageNet 终端分类”拆成两个阶段。**
当前最重要的问题不是“coeff 再调多大”，而是现有 trial schedule 和主 endpoint 尚未
严格对齐论文；同时，退化实验也尚未让 prime 向 target 提供可持续、额外且正确的信息。

第一优先级是精确实现论文的 clean no-prime/1-back/2-back 双 preview 序列，并对隐藏层
做五大类 time-resolved linear decoding。其后，最有希望、也最可证伪的 terminal
performance 修改是：

> clean same-image prime → degraded target，并让高层 PCoder state 在 pair 内持续；
> 以 clean different-image prime、no state carry 和 no-Hebb 为控制。

它同时满足两个必要条件：prime 的信息质量高于 target，而且该信息能真正进入 target
的 recurrent inference。

## 10. 推荐的分阶段实验

### 10.0 Stage P：paper-matched clean benchmark

先实现两个连续左右双物体 preview，再呈现中央 target；比较 no-prime、1-back、2-back。
prime 与 target 为完全相同图片，其他 preview 图片来自 target 之外的大类，不人为降低
prime 对比度或额外 blur。主 endpoint 是 Conv4/Conv5、FC1/FC2 的五类交叉验证线性
解码与 activation timeline：目标表型为 1-back activation↓ + decoding↑，且 2-back
更弱。timestep 暂不直接映射到论文的 150–210 ms。

Stage P 的刺激、统计、消融与记录规范见
[`biorxiv_654463_alignment.md`](biorxiv_654463_alignment.md)。只有完成该阶段后，
ImageNet Top-1 才作为论文机制向鲁棒识别外推的次级 endpoint。

三个后续阶段与 Stage P 组成单一证据链，而不是互斥方案：

```text
Stage P：clean no/1-back/2-back，复现论文神经表型
  → Stage A：target-only clean-reference diagnostic，确认 PCoder 是否恢复信息
  → Stage B：Same/Different clean prime × memory carrier 因果 pilot
  → Stage C：held-out terminal confirmation
  → Stage D：必要时才升级 decoder
```

Stage P 主要决定 Hebbian/inhibition 机制能否产生 suppression + hidden decoding
sharpening；Stage A 决定 PCoder carry 是否有可信的生成恢复基础；Stage B 把二者放进
同一个 1-back degraded-target trial 中，分离 Hebb-only、PCoder-carry-only 与协同效应。

### 10.1 Stage A：clean-reference generative diagnostic

先不做新的 priming 结论，只回答 decoder 是否在恢复信息。

首轮条件：noise 0.5、occlusion 0.3；200 odd targets；no-Hebb、无跨 trial PCoder
carry。若 clean-reference recovery 方向明确，再扩展到 noise 0.75、occlusion 0.4 和
1000 held-out targets。

每个 timestep 额外保存：

- PCoder1 pixel-space prediction 到 clean image 的距离；
- PCoder1–5 prediction 与 clean lower-layer target 的 MSE/cosine；
- degraded→recurrent representation 到 clean representation 的距离；
- true-class logit margin、wrong→right 与 right→wrong transitions；
- 内部 degraded-target prediction error，作为对照而非恢复指标。

正式结果保存逐样本标量距离；完整 PCoder1–5 tensors 只保存固定的小型 diagnostic
subset。若 clean-reference distance 不下降，就没有理由期望现有 decoder 支持生成式
priming。

### 10.2 Stage B：200-target causal pilot

选择 noise 0.5 与 occlusion 0.3，并使用 paper-matched 1-back sequence：preview-1 为
distractors，preview-2 含 clean parafoveal prime，中央 target 才退化。

| memory carrier | Same prime | Different prime | 目的 |
|---|---:|---:|---|
| 无 Hebb、PCoder reset | ✓ | 可复用一致结果 | target-only baseline |
| Hebb only，PCoder reset | ✓ | ✓ | 当前 channel memory，排除 prime 质量问题 |
| PCoder carry，no-Hebb | ✓ | ✓ | 直接检验生成式/semantic state carry |
| Hebb + PCoder carry | ✓ | ✓ | 检验两种 memory 是否协同 |

共约 14 个轻量条件：两个 degradation 的 no-memory baseline，加上三种有效 carrier ×
Same/Different × 两个 degradation。PCoder carry 指 preview→target 不 reset
`rep/prd/grd`，但在每个完整 trial 开始前 reset；Same/Different prime 使用相同图像
质量和时序，只有 identity 不同。

Stage B 不作为正式显著性结论，只做按以下顺序的 go/no-go：

- clean-reference feature distance 改善；
- 五类 hidden-state decoding 与 early decision dynamics 同时改善；
- Same−Different Top-1 再作为最强的次级判据；
- wrong→right 增加不能以更大的 right→wrong 或 clean accuracy 损失为代价；
- wrong-prime carry 必须明显差于 same-prime carry，否则只是通用 warm start。

### 10.3 Stage C：held-out confirmation

仅确认 Stage B 中预先选定的一个 memory carrier：

- targets：noise 0.5/0.75、occlusion 0.3/0.4；
- 1000 held-out targets；至少多个 degradation seeds；
- Same clean prime、Different clean prime、No prime、no-Hebb/无 carry causal control；
- primary paper endpoint：hidden decoding 的 Same−Different 为正，且 activation/
  suppression 方向与预先假设一致；
- recovery endpoint：clean-reference distance 相对 Different 和 No-prime 均改善；
- 若声称 terminal benefit，再要求 paired Top-1 Same−Different > 0、Same−NoPrime > 0，
  且两者 95% CI 均不跨 0；
- σ=2 不作为寻找正效应的条件，因为已处于 classification floor。

调参集与确认集必须分开，避免从同一 1000 targets 中选择最佳 timestep/参数后再报告
同一 CI。

### 10.4 Stage D：如果 PCoder carry 仍失败

这时才升级生成能力：

1. 用 corrupt high-level representation 预测 **clean lower-level representation**，
   而不是仅重建当前 degraded target；
2. 将单层 transpose convolution 替换为 residual upsample decoder，减少有限 receptive
   field 与反卷积伪影；
3. 为 PCoder5 增加 top semantic latent/attractor，使最高层有真正 prior；
4. 在 clean-reference diagnostic 上选择 feedback/error gains，再冻结参数做 held-out
   classification；
5. 逐层 ablation PCoder4、PCoder5 和 Hebbian boost，确认收益来自生成恢复而不是
   偶然 logit calibration。

## 11. 优先级判断

推荐顺序：

```text
paper-matched clean no/1-back/2-back + hidden-state MVPA
  → clean-reference diagnostic
  → clean prime / degraded target
  → pair-internal PCoder state carry
  → held-out classification confirmation
  → 最后才重训或扩容 decoder
```

原因是，当前最可能的瓶颈首先是“prime 信息没有进入 target 的生成状态”，其次才是
“生成器本身容量不足”。直接换更强 decoder 会同时改变模型容量、训练目标和 recurrent
dynamics，难以解释最终正效应究竟来自哪里。

## 12. 数据入口

- 自动分类报告：`Hebbian-CNN/Log/comparisons_v1/REPORT.md`
- 110 行时间线：`predify_vs_vanilla_timeline.csv`
- foveated classification：`foveated_priming_accuracy.csv`
- layout interaction：`foveated_vs_parafovea.csv`
- scale-invariant sharpening：`foveated_sharpening_t9.csv`
- priming 的 t0/t1/t9 探索性时间线：`priming_timeline_selected.csv`
- 原实验协议与 schema：`docs/vanilla_predify_timeline_foveated_v1.md`
- 主要 MEG 论文逐项对标与下一阶段规范：`docs/biorxiv_654463_alignment.md`
