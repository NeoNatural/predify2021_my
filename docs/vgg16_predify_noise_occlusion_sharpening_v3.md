# 原始 VGG16 FC 的 Predify 噪声/分形遮挡实验：方案改进、记录结构与复现说明

> 文档版本：2026-08-31  
> 对应结果 schema：v3  
> 对应正式结果目录：`Hebbian-CNN/Log/sharpening_v3/`

## 1. 文档目的

本文完整记录这次基于 `Hebbian-CNN/Hebb_VGG_Pred_Run.py` 完成的实验改进，重点回答以下问题：

1. 为什么要从原来的 `trace_var` 判据升级到归一化类内方差和 Fisher separability；
2. 如何构造 degradation-matched 的无 Hebbian 基线；
3. 原始 torchvision VGG16 FC、Predify PCoder、Hebbian 层、噪声和遮挡条件如何组合；
4. 40 个正式实验条件如何组织、命名和保存；
5. 每个 `.pckl` 文件和汇总 CSV 中具体保存了什么；
6. 如何复现实验、验证结果和避免误读。

这份文档描述的是已经完成并校验过的正式运行，而不只是计划。完整实验共包含 40 个 schema-v3 结果文件，总体积约 3.07 GiB。

## 2. 相比旧方案的关键改进

| 旧方案中的问题 | 本次改进 | 直接收益 |
|---|---|---|
| FC 权重来源可能混入单独微调 checkpoint | 固定 `fc_weight_source = "torchvision"`，使用 `VGG16_Weights.IMAGENET1K_V1` 原始 classifier 权重 | 结果可以明确解释为原始 torchvision VGG16 FC 上的效应 |
| 运行参数散落在脚本中 | 使用 TOML 配置、统一 CLI 和配置快照 | 每个结果都能追溯完整模型、刺激、seed 和路径设置 |
| 只有干净输入或单一退化条件 | 增加 Predify 式标准化张量高斯噪声和分形遮挡 | 可以检验 repetition effect 在输入不确定性下是否出现或消失 |
| `trace_var` 会随整体 activation 缩放而下降 | 增加 L2-normalized within-class variance | 将“整体激活降低”与“表征方向真正变集中”分离 |
| 只看类内集中程度，无法判断类别是否一起塌缩 | 增加 normalized between-class scatter 和 normalized Fisher ratio | 可以检验类内收缩是否伴随类别可分性保持或增强 |
| 缺少相同退化下的无 Hebbian 对照 | 增加 `original_mode = true` 的 no-Hebb 配置 | 排除 Predify recurrence、退化函数、trial schedule 和数值流程本身产生的 Prime–NonPrime 差异 |
| Prime–NonPrime 差异可能混入基线伪差 | 计算 `(Prime−NonPrime)_Hebb − (Prime−NonPrime)_noHebb` | 得到 Hebbian × repetition 的差分中的差分 |
| 旧结果难以扩展且字段语义不够集中 | 结果 schema 升级为 v3，并新增独立分析 CSV | 原始数据、在线统计、准确率、路径和交互效应可以统一审计 |
| σ=0 与已有干净实验重复 | 正式 v3 只运行五个非零噪声档位 | 避免重复计算；旧干净结果继续作为历史基线保存 |

## 3. 科学问题与判据

这里将“完整 repetition sharpening”拆成三个必须同时满足的部分：

1. **repetition suppression**：Prime 条件的平均 activation 更低；
2. **scale-invariant concentration**：Prime 条件的 L2 归一化类内方差更低；
3. **class separability**：Prime 条件的归一化 Fisher ratio 更高。

只有三个方向同时成立，才将该层标为 complete sharpening。原始 `trace_var` 仍保存在 v3 文件中以兼容旧分析，但不再参与严格判定。

### 3.1 为什么原始 `trace_var` 不够

若表征只发生统一缩放：

\[
x' = \alpha x,\qquad 0 < \alpha < 1,
\]

则 covariance 和 trace variance 自动变为：

\[
\Sigma' = \alpha^2\Sigma,
\qquad
\operatorname{tr}(\Sigma')=\alpha^2\operatorname{tr}(\Sigma).
\]

所以 `activation ↓ + raw trace_var ↓` 可能只是同一个 suppression 现象的两种表现，不能单独证明表征几何更精确。

### 3.2 L2-normalized within-class variance

先对每个表征向量归一化：

\[
\hat{x}_i=\frac{x_i}{\lVert x_i\rVert_2},
\]

再对类别 \(c\) 计算 biased covariance trace：

\[
W_c=rac{1}{n_c}\sum_{i\in c}
\lVert\hat{x}_i-\hat{\mu}_c\rVert_2^2.
\]

如果 Prime 只是把所有向量乘以常数，归一化后向量不变，因此 `normalized_trace_var` 不会下降。它只对方向和几何结构变化敏感。

对单位向量，该指标与平均类内 cosine similarity 有解析关系：

\[
W_c=\frac{n_c-1}{n_c}
\left(1-\overline{\cos}_{c}\right).
\]

因此 v3 同时记录 `normalized_trace_var` 和 `within_cosine`，二者可以互相校验。

### 3.3 Normalized Fisher separability

对所有有效类别中心，定义：

\[
S_B=\frac{1}{C}\sum_c
\lVert\hat{\mu}_c-\hat{\mu}\rVert_2^2,
\qquad
S_W=\frac{1}{C}\sum_c W_c,
\]

并计算：

\[
J=\frac{S_B}{S_W}.
\]

`normalized_fisher_ratio ↑` 表示类别中心间结构相对于类内离散程度得到改善。这样可以排除“所有类别一起向同一点塌缩”却被错误解释为 sharpening 的情况。

### 3.4 效应量和差分中的差分

每个 Prime–NonPrime 相对效应定义为：

\[
E_M=100\left(\frac{M_{Prime}}{M_{NonPrime}}-1\right)\%.
\]

其中 \(M\) 分别为 activation、normalized variance 或 normalized Fisher。随后计算：

\[
I_M=E_{M,Hebb}-E_{M,noHebb}.
\]

最终严格交互判据为：

\[
I_{activation}<0,
\qquad
I_{normalized\ variance}<0,
\qquad
I_{normalized\ Fisher}>0.
\]

## 4. 完整实验设计

### 4.1 条件矩阵

| 维度 | 设置 |
|---|---|
| 模型状态 | active Hebbian；no-Hebb pass-through |
| repetition 条件 | Prime；NonPrime |
| 输入退化 | Predify noise；fractal occlusion |
| 噪声强度 | σ = 0.25、0.5、0.75、1.0、2.0 |
| 遮挡强度 | intensity = 0.1、0.2、0.3、0.4、0.5 |
| 每个条件样本数 | 2000 trials |
| 正式结果总数 | 10 degradation points × 2 repetition conditions × 2 model states = 40 files |

σ=0 和 intensity=0 都是 identity degradation。本次 v3 没有重复运行 0 档；此前 schema-v2 干净实验继续作为历史干净基线，但不能提供 v3 的 Fisher 字段。

### 4.1.1 四份正式配置共有的关键值

| 配置项 | 正式值 | 说明 |
|---|---|---|
| `paths.sample_file` | `Hebbian-CNN/Log/DatasetSample/N_Back-1.pckl` | 固定 trial schedule |
| `paths.pcoder_weight_dir` | `predify2021/model_factory` | 五组本地预训练 PCoder 权重 |
| `paths.fc_checkpoint` | 空 | 不加载 FC 微调 checkpoint |
| `runtime.device` | `auto`（正式命令覆盖为 `cuda`） | 正式结果 metadata 为 CUDA |
| `runtime.seed` | 114514 | Python、NumPy、Torch 和 CUDA 全局 seed |
| `runtime.max_time_steps` | 10 | odd/single target timestep 数 |
| `runtime.reduced_time_steps` | 2 | even/half-field timestep 数 |
| `runtime.max_samples` | 2000 | 每个独立文件的 trial 数 |
| `model.name` | `pvgg` | Predify VGG16 |
| `model.pretrained_pcoders` | `true` | 加载五组 PCoder 权重 |
| `model.random_init` | `true` | 保留当前 PVGG 构造参数；实际运行值写入 metadata |
| `model.fc_weight_source` | `torchvision` | 原始 VGG16 FC 来源 |
| `model.fc_hebb_compute_enabled` | `true` | FC wrapper 路径启用；实际更新仍受 layer coeff 控制 |
| `stimulus.canvas_scale` | 2.0 | half-field canvas 宽度比例 |
| `stimulus.contrast_scale` | 0.5 | half-field 单图对比度比例 |
| `stimulus.degradation_seed` | 0 | 退化基础 seed |
| `stimulus.fractal_octaves` | 3 | 分形随机场 octave 数 |
| `stimulus.blur_sigma` | 1.0 | half-field 历史 Gaussian blur |
| `stimulus.downsample_after_blur` | `true` | blur 后缩小 0.5 倍 |
| `experiment.task_name` | `N_Back-` | 输出命名前缀 |
| `experiment.gap` | 1 | N-back gap |
| `experiment.run_priming` | `true` | 运行 Prime |
| `experiment.run_nonpriming` | `true` | 运行 NonPrime |
| `experiment.representation_class_items` | 400 | 用前 400 个 schedule item 确定 200 个表征类 |

### 4.2 模型与权重

- 模型名称：`pvgg`；
- backbone/classifier 来源：torchvision VGG16 `IMAGENET1K_V1`；
- FC checkpoint：不加载，`fc_checkpoint = ""`；
- PCoder：使用仓库中已有的五组预训练权重；
- 五个 PCoder 超参数均为 `ffm=0.8`、`fbm=0.1`、`erm=0.01`；
- 推理模式：`model.eval()` 与 `torch.no_grad()`；
- 全局 seed：114514；退化 seed：0。

四个插入并记录的表征位置如下：

| 保存名 | 文档简称 | 位置 | 默认 Hebbian 参数 |
|---|---|---|---|
| `hebb_pcoder4` | Conv4 | PCoder4 representation | `decay=0.5, coeff=0.05, inh_c=4` |
| `hebb_pcoder5` | Conv5 | PCoder5 representation | `decay=0.5, coeff=0.05, inh_c=4` |
| `hebbian_1` | FC1 | VGG classifier 第一处 Hebbian wrapper | `decay=0, coeff=0, cut_perc=0` |
| `hebbian_2` | FC2 | VGG classifier 第二处 Hebbian wrapper | `decay=0, coeff=0, cut_perc=0` |

需要特别注意：当前默认参数下，真正具有非零 Hebbian 更新系数的是 Conv4/Conv5。FC1/FC2 仍被 wrapper 包装、启用和记录，但其 `coeff=0`，所以 FC 层观测到的 repetition 差异主要反映上游 Conv Hebbian 状态向 classifier 的传播，而不是 FC 自身发生非零权重更新。

active 与 no-Hebb 的唯一本质差异是：

- active：`original_mode = false`；
- no-Hebb：`original_mode = true`，四个 wrapper 直接 pass-through，不建立跨 trial 的 Hebbian boost。

no-Hebb 仍保留 torchvision FC、五个 Predify PCoder、recurrent timestep、输入退化、seed 和 trial schedule，因此是 degradation-matched 基线，不是普通静态 VGG 基线。

### 4.3 Trial 序列与 timestep

样本序列来自：

`Hebbian-CNN/Log/DatasetSample/N_Back-1.pckl`

2000 个 trial 按 0-based index 交替组织：

```text
trial 0：half-field prime/control stimulus，2 timesteps，pair 开始时 reset Hebbian state
trial 1：single target stimulus，10 timesteps，记录 repeated-trial accuracy 和 representation
trial 2：下一组 half-field stimulus，2 timesteps，reset
trial 3：下一组 single target，10 timesteps
...
```

Prime 与 NonPrime 的 odd/single target 序列相同。差别发生在前一个 even/half-field trial：

- Prime 使用当前 schedule 指定的 priming image；
- NonPrime 将 priming image 替换为 `idx_log_list[(trial_idx + 2) % len(idx_log_list)]`；
- half-field 的另一个图像来自 `trial_idx + 4` 的 schedule 位置。

`acc_result` 只统计 `acc_top_1[1::2]` 和 `acc_top_5[1::2]`，即 1000 个 odd/single repeated trials。even trial 的分类真假值不进入最终条件准确率。

even trial 实际只运行 2 timesteps。`Out_list` 的 timestep 2–9 会复制该 trial 的最后输出，以维持统一 `[N,T,1000]` 形状；`layer_metrics_arr` 未运行的位置保留初始化值。正式 sharpening 分析只读取 odd trials，所以不会使用这些 even-trial padding。

### 4.4 表征样本设计

`representation_class_items = 400` 从 sample schedule 前 400 项确定表征类别集合，实际得到 200 个类别。在线统计只在 odd/single trial 更新。

在 2000-trial 全量运行中：

- odd trials 共 1000 个；
- 表征类别共 200 类；
- 每类正好 5 个表征样本；
- 每个 timestep 和每个记录层都独立累计统计。

这也是为什么只有几条样本的普通 smoke test 无法得到有限 Fisher：每个类别至少要出现两次，类内散度才有定义。本次 GPU 集成冒烟采用过 800 samples，使每类有两次 odd 表征；正式结果则使用每类五次。

## 5. 输入退化的精确定义

### 5.1 Predify 式标准化张量高斯噪声

噪声直接加在 torchvision ImageNet transform 输出的标准化 tensor 上：

\[
x_{noisy}=x+\sigma z,
\qquad z\sim\mathcal{N}(0,I).
\]

当前实现每次用 `degradation_seed=0` 重新建立 CPU generator。对于相同 tensor shape：

- 不同 σ 使用同一个标准高斯 realization，只改变缩放系数；
- Prime/NonPrime 使用配对 realization；
- active/no-Hebb 也使用配对 realization；
- 由于 generator 每次从相同 seed 开始，同尺寸的不同图像也会叠加相同空间噪声模板。

最后一点有利于严格配对，但也是设计限制：本实验检验的是固定噪声 realization 下的效应，而不是对多种噪声样本取期望。

### 5.2 分形遮挡

分形遮挡不添加高斯噪声。其流程为：

1. 生成 3 个 octave 的多尺度随机场；
2. 计算图像 edge map；
3. 用 `fractal_mask × (1 + 2.5 × edge_map)` 得到 edge-guided signal；
4. intensity 控制二值化阈值；
5. 用 average pooling 将 mask 软化；
6. 将遮挡区域混合为 ImageNet 标准化空间中的黑色值 `-mean/std`。

运行器对分形遮挡使用 `degradation_seed + sample_index`。因此每个样本有自己的随机场，而同一 sample index 在不同 intensity、Prime/NonPrime 和 active/no-Hebb 条件下保持配对。不同 intensity 通过同一底层场的阈值变化产生强弱遮挡。

### 5.3 与历史 prime blur 的顺序

对于 even/half-field stimulus，两种 repetition 条件都遵循：

```text
对两张图分别调低 contrast
→ 拼成双视野 canvas
→ 应用 noise 或 fractal occlusion
→ Gaussian blur（sigma=1.0）
→ bilinear 下采样 0.5 倍
```

odd/single target 只应用输入退化，不应用 half-field blur/downsample。输入退化与历史 prime blur 是两个独立操作，顺序固定为“先退化、后 blur”。

## 6. 配置文件与代码职责

### 6.1 四个正式配置

| 文件 | 退化 | Hebbian 状态 | 默认输出目录 |
|---|---|---|---|
| `configs/hebb_vgg_predify_noise.toml` | Predify noise | active | `.../sharpening_v3/hebb` |
| `configs/hebb_vgg_fractal_occlusion.toml` | fractal occlusion | active | `.../sharpening_v3/hebb` |
| `configs/hebb_vgg_predify_noise_nohebb.toml` | Predify noise | pass-through | `.../sharpening_v3/nohebb` |
| `configs/hebb_vgg_fractal_occlusion_nohebb.toml` | fractal occlusion | pass-through | `.../sharpening_v3/nohebb` |

相对路径均以 TOML 中 `[project].root` 解析，而不是依赖启动命令的当前目录。命令行可以覆盖 ImageNet root、输出目录、device 和样本数，最终解析后的配置会完整写入每个结果的 `run_metadata.config`。

### 6.2 核心代码

| 文件 | 职责 |
|---|---|
| `Hebbian-CNN/Hebb_VGG_Pred_Run.py` | 历史兼容入口，默认指向专用 noise 配置 |
| `predify_experiments/cli.py` | 统一命令行入口和参数覆盖 |
| `predify_experiments/config.py` | TOML 读取、路径解析和运行前验证 |
| `predify_experiments/hebb_vgg_priming.py` | 模型构建、trial loop、hook、准确率和 schema-v3 序列化 |
| `predify_experiments/sequence.py` | recurrent sequence、Hebbian update 延迟提交和 reset |
| `Hebbian-CNN/Hebb_Priming_Utils.py` | half-field stimulus、noise、fractal occlusion、blur 和下采样 |
| `Hebbian-CNN/OnlineSharpeningStats_Demo.py` | GPU 在线类内/类间表征统计 |
| `predify_experiments/sharpening_analysis.py` | Prime 配对、no-Hebb 配对、差分中的差分和 CSV 输出 |
| `tests/test_online_sharpening_stats.py` | scaling invariance、cosine identity、Fisher 分离测试 |
| `tests/test_sharpening_analysis.py` | legacy normalized variance 恢复和严格判据测试 |
| `tests/test_config.py` | 四份正式配置的约束测试 |
| `tests/test_stimulus_degradation.py` | noise、fractal、zero identity 和处理顺序测试 |

## 7. 正式结果目录与文件结构

```text
Hebbian-CNN/
└── Log/
    ├── DatasetSample/
    │   └── N_Back-1.pckl                 # 输入 trial schedule，不属于 v3 输出
    └── sharpening_v3/
        ├── hebb/                          # active Hebbian，20 个 pckl，约 1.53 GiB
        │   ├── ...predify_noise_sigma0p25_Prime.pckl
        │   ├── ...predify_noise_sigma0p25_nonPrime.pckl
        │   ├── ...predify_noise_sigma0p5_Prime.pckl
        │   ├── ...                        # 五个 σ × 两种 repetition 条件
        │   ├── ...fractal_occlusion_intensity0p1_Prime.pckl
        │   └── ...fractal_occlusion_intensity0p5_nonPrime.pckl
        ├── nohebb/                        # matched no-Hebb，20 个 pckl，约 1.53 GiB
        │   └── 与 hebb/ 完全相同的 20 个文件名
        ├── hebbian_interaction_t9.csv     # 40 行 layer-level 汇总
        └── REPORT.md                      # 简短结果摘要
```

整个 `Hebbian-CNN/Log` 被 `.gitignore` 排除，因此这些 3.07 GiB 实验产物不会进入 Git。代码与配置可以通过版本控制恢复，但正式 `.pckl`、CSV 和报告需要单独备份。

### 7.1 文件命名规则

通用格式：

```text
VGG_Result_N_Back-1_new_{degradation}_{level}_{condition}.pckl
```

具体格式：

```text
VGG_Result_N_Back-1_new_predify_noise_sigma{level}_{Prime|nonPrime}.pckl
VGG_Result_N_Back-1_new_fractal_occlusion_intensity{level}_{Prime|nonPrime}.pckl
```

小数点用 `p` 替代，例如：

- `sigma0p25` = σ 0.25；
- `sigma1` = σ 1.0；
- `intensity0p4` = intensity 0.4。

`hebb/` 和 `nohebb/` 内使用相同文件名；目录本身表示模型状态。这样分析器可以仅通过文件名一一匹配四个源文件：Hebb Prime、Hebb NonPrime、no-Hebb Prime、no-Hebb NonPrime。

## 8. 单个 schema-v3 `.pckl` 的内容

下面的 shape 来自实际 2000-sample 正式文件，而不是估算。

| 顶层字段 | 类型/shape | 含义 |
|---|---|---|
| `schema_version` | `int`，值为 3 | 结果格式版本 |
| `run_metadata` | `dict` | 时间、配置文件、device、seed、condition、退化、Hebbian 状态和完整配置快照 |
| `task_name` | `str` | 当前为 `N_Back-` |
| `gap` | `int` | 当前为 1 |
| `max_time_step` | `int` | 当前为 10 |
| `idx_log_list` | list `[2000]` | 本次实际使用的 ImageNet sample indices |
| `full_clss_list` | list `[2000]` | sample schedule 中对应的 class sequence |
| `y_list` | list `[2000]` | runner 使用的 trial labels；NonPrime even trial 可为 -1 |
| `Out_list` | ndarray `[2000,10,1000]` | 每个 trial、timestep 的 VGG logits |
| `layer_metrics_arr` | ndarray `[2000,10,4,6]` | 四层 activation/sparsity 分布指标 |
| `pcoder_error_arr` | ndarray `[2000,10,2,1]` | PCoder4、PCoder5 prediction error |
| `pcoder_indices_to_record` | list `[4,5]` | `pcoder_error_arr` 最后一层索引语义 |
| `repr_layer_names` | list `[4]` | 表征层顺序 |
| `repr_class_subset` | list `[200]` | 在线表征统计使用的原始类别标签 |
| `repr_stats_timeline` | list `[10]` | 每个 timestep 的四层在线表征统计 |
| `repr_stats` | dict | `repr_stats_timeline[-1]` 的便捷别名 |
| `acc_top_1` | list `[2000]` of bool | 每个 trial 的最终 timestep Top-1 是否正确 |
| `acc_top_5` | list `[2000]` of bool | 每个 trial 的最终 timestep Top-5 是否正确 |
| `acc_result` | tuple `(top1,top5)` | 只对 `[1::2]` repeated trials 求得的条件准确率 |

### 8.1 `run_metadata`

关键字段包括：

| 字段 | 示例/含义 |
|---|---|
| `created_utc` | 文件生成 UTC 时间 |
| `config_file` | 使用的 TOML 绝对路径 |
| `device` | `cuda` |
| `seed` | 114514 |
| `condition` | `priming` 或 `nonpriming` |
| `degradation_mode` | `predify_noise` 或 `fractal_occlusion` |
| `degradation_level` | 当前 σ/intensity |
| `degradation_seed` | 0 |
| `hebbian_condition` | `hebb` 或 `nohebb` |
| `config` | `asdict(ExperimentConfig)` 的完整解析快照 |

配置快照中包含当时实际解析的 ImageNet 路径和输出路径，因此移动文件后，旧绝对路径仍表示运行时来源，不会自动更新。

### 8.2 `layer_metrics_arr`

layer axis 的实际顺序为：

```text
0: hebbian_2   (FC2)
1: hebbian_1   (FC1)
2: hebb_pcoder5 (Conv5)
3: hebb_pcoder4 (Conv4)
```

最后一个 metric axis 的顺序为：

```text
0: mean activation
1: non-zero fraction
2: maximum
3: 25% quantile
4: 50% quantile
5: 75% quantile
```

严格 sharpening 分析使用 odd trials、目标 timestep 和 metric index 0。

### 8.3 `repr_stats_timeline`

每个 timestep 下，每层保存：

| 字段 | shape | 含义 |
|---|---:|---|
| `count` | `[200]` | 每类观测数；正式结果最终均为 5 |
| `trace_var` | `[200]` | 未归一化类内 covariance trace，仅供兼容/诊断 |
| `normalized_trace_var` | `[200]` | L2-normalized 类内 covariance trace |
| `within_cosine` | `[200]` | 每类平均 pairwise cosine similarity |
| `within_scatter` | scalar | 有效类别 raw `trace_var` 均值 |
| `normalized_within_scatter` | scalar | 有效类别 normalized variance 均值 |
| `between_scatter` | scalar | raw class-centroid between scatter |
| `normalized_between_scatter` | scalar | normalized centroid between scatter |
| `fisher_ratio` | scalar | raw between/within ratio |
| `normalized_fisher_ratio` | scalar | scale-invariant between/within ratio |

在线累计使用 mixed precision：feature sum 与 normalized-vector sum 使用 bfloat16，平方和与 normalized squared norm 使用 float32。这控制了 GPU 内存占用，但 raw variance 在极小方差时仍可能出现 cancellation，因此科学判据以 normalized geometry 为主。

### 8.4 v2 与 v3 的兼容边界

旧 v2 文件若包含 `count` 和 `within_cosine`，可以通过单位向量恒等式恢复 `normalized_trace_var`。但旧文件没有保存 normalized class centroids 或 between scatter，所以无法可靠重建 Fisher ratio。

因此：

- 只重新画 normalized variance：旧 v2 可以部分复用；
- 需要 normalized Fisher 或严格三联判据：必须使用本次 v3 重跑结果；
- 不应在同一个 Fisher 汇总中混用 v2 与 v3。

## 9. 汇总 CSV 的结构

文件：

`Hebbian-CNN/Log/sharpening_v3/hebbian_interaction_t9.csv`

它包含 40 行：10 个 degradation points × 4 个 layers。`timestep=-1` 表示读取最后一个 timestep，即 t9。

字段可以分为五组：

1. **匹配键**：`degradation_mode`、`degradation_level`、`layer`、`timestep`；
2. **准确率**：Hebb/no-Hebb 各自的 Prime/NonPrime Top-1、Top-5；
3. **Hebb 内部效应**：activation、normalized trace、within cosine、normalized Fisher 和 complete flag；
4. **no-Hebb 内部效应**：同一组字段，用于检查基线伪差；
5. **交互效应**：三项 difference-in-differences、`interaction_complete_sharpening` 和四个绝对源文件路径。

CSV 中每层重复准确率字段是有意设计：这样任意抽取一行时仍是自包含的，不需要再 join 另一张 condition-level 表。

## 10. 复现实验

以下命令从仓库根目录运行。正式环境为 `predify2021` Conda 环境，ImageNet 路径在命令行覆盖配置占位符。

### 10.1 Active Hebbian：Predify noise

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_predify_noise.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" `
  --device cuda `
  --max-samples 2000
```

### 10.2 Active Hebbian：fractal occlusion

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_fractal_occlusion.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" `
  --device cuda `
  --max-samples 2000
```

### 10.3 Matched no-Hebb：Predify noise

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_predify_noise_nohebb.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" `
  --device cuda `
  --max-samples 2000
```

### 10.4 Matched no-Hebb：fractal occlusion

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_fractal_occlusion_nohebb.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" `
  --device cuda `
  --max-samples 2000
```

### 10.5 生成交互汇总

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments.sharpening_analysis `
  --hebb-dir Hebbian-CNN/Log/sharpening_v3/hebb `
  --nohebb-dir Hebbian-CNN/Log/sharpening_v3/nohebb `
  --output-csv Hebbian-CNN/Log/sharpening_v3/hebbian_interaction_t9.csv
```

### 10.6 测试

```powershell
conda run --no-capture-output -n predify2021 `
  python -m unittest discover -s tests -v
```

当前验证结果为 34/34 tests passed（包含后续 vanilla timeline 与 foveated
layout 补充测试）。

## 11. 已完成的数据完整性校验

正式结果已经逐文件检查：

- `hebb/` 20 个 `.pckl`；
- `nohebb/` 20 个 `.pckl`；
- 40/40 文件可以成功反序列化；
- 40/40 文件 `schema_version == 3`；
- 40/40 文件包含 2000 个 sample indices 和 2000 个 labels；
- 所有层、所有最终 timestep 的 200 个类别 count 均为 5；
- 所有最终 normalized Fisher ratio 均为有限值；
- 汇总 CSV 恰好 40 行；
- 21/40 个 degradation×layer 行满足严格 interaction complete criterion；
- Hebb/no-Hebb 各 20 个文件，metadata 中 `hebbian_condition` 正确；
- no-Hebb Prime/NonPrime 准确率在每个 degradation point 完全相同。

no-Hebb 的最大绝对数值伪差为：

- activation：`3.08×10^-6 %`；
- normalized variance：`1.10×10^-5 %`；
- normalized Fisher：`5.30×10^-5 %`。

这些微小差异属于独立 GPU 累计时的数值误差量级，因此交互效应实际上等同于 active Hebbian repetition effect。

## 12. 正式结果摘要

| 退化 | Level | Hebb Prime Top-1/5 | Hebb NonPrime Top-1/5 | no-Hebb Top-1/5 | 完整层 |
|---|---:|---:|---:|---:|---|
| Predify noise | 0.25 | 59.7 / 81.6 | 60.1 / 81.9 | 60.4 / 82.1 | Conv4、Conv5 |
| Predify noise | 0.50 | 38.2 / 62.7 | 38.4 / 62.6 | 38.7 / 63.0 | Conv4、Conv5 |
| Predify noise | 0.75 | 16.8 / 35.1 | 16.8 / 36.0 | 18.1 / 37.1 | Conv4、Conv5 |
| Predify noise | 1.00 | 6.7 / 15.5 | 6.7 / 16.2 | 6.8 / 17.1 | Conv4、Conv5 |
| Predify noise | 2.00 | 0.7 / 1.6 | 0.8 / 1.7 | 0.7 / 1.9 | 四层，但属于 floor |
| Fractal occlusion | 0.10 | 70.3 / 89.2 | 70.5 / 89.4 | 70.5 / 89.3 | Conv4、Conv5 |
| Fractal occlusion | 0.20 | 66.7 / 88.1 | 67.2 / 88.2 | 67.3 / 88.4 | Conv4、Conv5 |
| Fractal occlusion | 0.30 | 60.7 / 80.9 | 60.9 / 81.1 | 60.6 / 80.8 | Conv4、Conv5 |
| Fractal occlusion | 0.40 | 47.0 / 70.1 | 46.7 / 70.0 | 47.0 / 69.7 | Conv4、Conv5 |
| Fractal occlusion | 0.50 | 38.7 / 58.4 | 38.5 / 57.4 | 38.7 / 59.0 | 仅 Conv5 |

主要解释如下：

1. **非地板噪声 σ=0.25–1.0**：Conv4 和 Conv5 稳定满足三联判据；Conv5 效应更明显。
2. **分形遮挡 0.1–0.4**：Conv4/Conv5 的符号均完整，但 Conv4 normalized variance 的收缩只有约 0.006–0.009%，实际效应很小。
3. **遮挡 0.5**：Conv4 normalized variance 转为增加；Conv5 只保留很弱的完整效应。因此“强遮挡大体破坏 sharpening”的旧结论仍成立，但存在小的 Conv5 residual。
4. **FC1**：大部分非地板条件下 normalized variance 下降、Fisher 上升，但 activation 上升；属于 concentration without suppression。
5. **FC2**：大部分非地板条件下 activation 下降，但 normalized variance 上升；属于 suppression without sharpening。
6. **σ=2**：四层形式上都满足方向判据，但 Top-1 低于 1%，应视为 representation/accuracy floor，而不能视为更强的有效 sharpening。
7. **分类性能**：active Hebbian 没有表现出一致的准确率优势；局部几何 sharpening 与最终分类准确率改善不是同一件事。

因此最稳妥的总体结论是：

> 噪声和中等强度遮挡可以在 Conv4/Conv5 暴露出 scale-invariant 的局部 repetition suppression + sharpening，但该效应没有完整传播为 FC1/FC2 上一致的全网络效应，也没有转化为稳定的分类性能提升。

## 13. 限制与后续建议

### 13.1 当前限制

- 所有退化只使用一个 deterministic seed；
- noise 对同尺寸不同图像使用同一空间 realization；
- 每类只有 5 个 repeated representations；
- Fisher 是每层/每 timestep 的 aggregate scalar，目前没有直接保存 per-sample bootstrap distribution；
- Conv4 和高遮挡条件中的 normalized variance 效应很小，只能作描述性结果；
- σ=2 存在明显 accuracy floor；
- FC1/FC2 默认 Hebbian `coeff=0`，不能将其表征变化解释为 FC 自身 Hebbian 更新；
- schema-v3 pckl 总体积较大且被 Git 忽略，需要额外归档策略。

### 13.2 推荐后续实验

1. 增加多个 degradation seeds，并报告跨 seed 均值与置信区间；
2. 对类别或 trial 做 bootstrap，给 normalized variance 和 Fisher interaction 添加 95% CI；
3. 将 noise seed 改为 `base_seed + sample_index`，同时在不同 σ 间保持配对，以检验跨噪声 realization 的稳健性；
4. 将 σ=2 单独标记为 stress/floor condition，不与 σ=0.25–1.0 合并推断；
5. 若要研究 FC 本地 Hebbian sharpening，应单独设计非零 FC `coeff` 条件，并与当前 upstream-only 传播结果区分；
6. 保存更轻量的 analysis-ready per-class/per-trial statistics，减少必须长期保留全部 logits 的存储压力；
7. 对 clean v3 条件只补跑一个 σ=0 matched matrix，而不是重新引入大量 0 档，以获得完全同 schema 的干净 Fisher 基线。

## 14. 结果维护注意事项

- 不要将新结果直接写入旧 schema-v2 文件所在目录；
- 修改 degradation levels 时，应同时修改 active/no-Hebb 两份配置；
- 分析前确认四个 matched 文件都存在，分析器会在缺文件时直接报错；
- 重新运行同名条件会覆盖现有 `.pckl`，运行前应备份正式目录；
- `run_metadata.config` 是运行时配置真值，文件名只用于快速定位；
- `REPORT.md` 是人工可读摘要，`hebbian_interaction_t9.csv` 是机器可读汇总，`.pckl` 才是可重新分析的原始实验记录；
- 由于 `Hebbian-CNN/Log` 被 `.gitignore` 排除，Git status 不会提示正式结果是否丢失或被覆盖。
