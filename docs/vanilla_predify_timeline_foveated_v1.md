# Vanilla VGG、Predify 时间线与 Foveated Priming 补充实验

## 1. 目的与证据边界

本补充实验回答两个在继续调节 Hebbian/PCoder 参数之前必须区分的问题：

1. **整个 Predify predictive-coding system 是否改善退化输入的分类？**
   使用 torchvision 原始 VGG16 作为真正的 feedforward 基线，并与 no-Hebb
   Predify 保存的 t0–t9 逐时间步配对比较。
2. **完整、居中的 foveated prime 是否比现有 parafoveal half-field prime
   更容易产生分类 priming？**
   在 clean、noise 0.5、fractal occlusion 0.3 三个代表点运行
   Hebb/no-Hebb × Prime/NonPrime。

Vanilla 对比检验的是加入整个 Predify 系统后的净效应，不能单独归因于
feedback、memory 或 error-correction。Foveated 对比固定现有模型参数，只改变
even prime 的布局与清晰度。

## 2. 为什么不能再把当前 t0 当作 vanilla feedforward

原始 2021 Predify 实现使用 `random_init=false`。当前仓库的正式实验配置使用
`random_init=true`；在这种初始化路径中，模型先用 backbone 建立 PCoder 状态，
随后才返回第一个被保存的输出。因此当前结果中的 t0 已包含 predictive update，
不是未接入 predictive coding 的 torchvision VGG16 输出。

本实验采用以下术语：

- **vanilla**：`torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)`
  的单次前向；
- **Predify t0**：当前 `random_init=true` 流程保存的第一个 predictive state；
- **Predify t1–t9**：其后 recurrent refinement states。

这样既不错误解释 t0，也不改变已有 parafovea 结果的初始化语义。

## 3. 样本、权重与共同设置

- ImageNet validation root：运行时通过 `--imagenet-root` 指定；
- schedule：`Hebbian-CNN/Log/DatasetSample/N_Back-1.pckl`；
- 正式 trial 数：2000，其中 odd target 为 1000 个；
- 1000 个 odd targets 由 200 类构成，每类 5 个样本；
- FC：torchvision `VGG16_Weights.IMAGENET1K_V1` 原始 classifier；
- Predify：仓库内五组预训练 PCoder 权重；
- Predify 初始化：`random_init=true`；
- even trial：2 timesteps；odd target：10 timesteps；
- 全局 seed：114514；degradation seed：0；
- bootstrap：样本级配对重采样 10,000 次，seed 114514。

## 4. Vanilla VGG16 benchmark

### 4.1 覆盖条件

Vanilla 只处理 schedule 的 odd targets，不建立跨 trial 状态：

| 模式 | levels |
|---|---|
| clean | 0.0 |
| Predify tensor noise | 0.25、0.5、0.75、1.0、2.0 |
| fractal occlusion | 0.1、0.2、0.3、0.4、0.5 |

共 11 个输入点、每点 1000 次单次 forward。输入 transform 与 Predify 实验均为
`VGG16_Weights.IMAGENET1K_V1.transforms()`。

### 4.2 输入配对

- noise 在 ImageNet 标准化张量上加 `N(0, σ)`；相同 shape 和 seed 共享同一
  标准高斯 realization；
- fractal mask 使用 `degradation_seed + original_trial_index`；这里的
  `original_trial_index` 是 schedule 中的 odd 位置 1、3、…、1999；
- dataset index、label、噪声或遮挡 seed 均与已有 Predify odd targets 对齐；
- vanilla 结果保存 `trial_indices`，用于显式验证 odd parity。

### 4.3 Vanilla 文件 schema

每个条件保存一个 `.pckl`，顶层字段为：

| 字段 | shape/类型 | 含义 |
|---|---:|---|
| `schema_version` | scalar | vanilla benchmark schema，当前为 1 |
| `benchmark_type` | string | torchvision VGG16 权重标识 |
| `run_metadata` | dict | 时间、设备、seed、退化与完整配置快照 |
| `idx_log_list` | `[1000]` | ImageNet dataset indices |
| `trial_indices` | `[1000]` | 原 schedule odd indices |
| `y_list` | `[1000]` | labels |
| `logits` | `[1000,1000]` | float32 logits |
| `acc_top_1` | `[1000]` | per-target Top-1 correctness |
| `acc_top_5` | `[1000]` | per-target Top-5 correctness |
| `acc_result` | tuple | aggregate Top-1/Top-5 |

Vanilla 文件不伪装成 priming schema，也不包含 PCoder、Hebbian 或 representation
字段。

## 5. Foveated full-image priming

### 5.1 布局定义

新增 `stimulus.prime_layout`：

- `parafoveal_half_field`：旧行为，也是旧配置缺省值；
- `foveated_full`：新增完整居中 prime。

`foveated_full` 的 even trial：

- Prime 使用当前 schedule index，即与随后 odd target 相同的原始 224×224 图像；
- NonPrime 使用 `idx_log_list[(trial_idx + 2) % len(idx_log_list)]`；
- 不拼接 second image；
- 不做 contrast reduction；
- 不做 historical Gaussian blur；
- 不做 downsample；
- noise/occlusion 仍使用共同的 `apply_input_degradation` 与 seed 规则。

odd target 不受 `prime_layout` 影响。测试直接验证相同参数下两个布局的 odd target
逐元素相等。

### 5.2 正式矩阵

代表点为 clean、noise 0.5、fractal occlusion 0.3。每点运行：

| 模型状态 | repetition condition |
|---|---|
| active Hebbian (`original_mode=false`) | Prime、NonPrime |
| no-Hebb (`original_mode=true`) | Prime、NonPrime |

共 12 个 schema-v3 文件。no-Hebb 保留 Predify PCoders，只让四个 Hebbian module
pass through；它不是 vanilla VGG。

### 5.3 Foveated schema-v3 记录

沿用既有 schema v3，包括：

- 2000×10×1000 logits timeline；
- labels、dataset indices、Top-1/Top-5 correctness 与 aggregate accuracy；
- 四个 Hebbian 层的 activation/nonzero/max/quantile timeline；
- PCoder prediction error timeline；
- 每个 timestep、每层、每类的 raw/normalized variance、within cosine；
- raw/normalized within scatter、between scatter 与 Fisher ratio；
- config、FC source、seed、degradation、Hebbian condition；
- 新增 `run_metadata.prime_layout = "foveated_full"`。

文件名显式包含 `_foveated_full`，不会覆盖旧 parafovea 文件。

## 6. 多 degradation 配置

旧 TOML 的单模式写法保持兼容：

```toml
[stimulus]
degradation_mode = "predify_noise"
degradation_levels = [0.25, 0.5]
```

新矩阵写法为：

```toml
[stimulus]
prime_layout = "foveated_full"

[[stimulus.degradations]]
mode = "none"
levels = [0.0]

[[stimulus.degradations]]
mode = "predify_noise"
levels = [0.5]

[[stimulus.degradations]]
mode = "fractal_occlusion"
levels = [0.3]
```

两种写法不能同时出现。配置验证会拒绝未知模式、负 level、重复 mode/level、
非零 `none` 和大于 1 的 fractal intensity。

## 7. 目录与文件结构

所有正式产物位于 `.gitignore` 排除的 `Hebbian-CNN/Log`：

```text
Hebbian-CNN/Log/
├── vanilla_vgg16_v1/
│   ├── Vanilla_VGG16_clean.pckl
│   ├── Vanilla_VGG16_predify_noise_sigma{...}.pckl       # 5
│   └── Vanilla_VGG16_fractal_occlusion_intensity{...}.pckl # 5
├── foveated_v1/
│   ├── hebb/       # 3 degradation points × Prime/NonPrime = 6
│   └── nohebb/     # 3 degradation points × Prime/NonPrime = 6
├── sharpening_v3/
│   ├── hebb/       # 既有 parafovea nonzero results
│   └── nohebb/
└── comparisons_v1/
    ├── predify_vs_vanilla_timeline.csv
    ├── foveated_priming_accuracy.csv
    ├── foveated_vs_parafovea.csv
    ├── foveated_sharpening_t9.csv
    ├── predify_vs_vanilla_timeline.png
    ├── foveated_vs_parafovea.png
    └── REPORT.md
```

Clean parafovea 布局比较读取旧 schema-v2 sigma=0 Prime/NonPrime；noise 0.5 和
occlusion 0.3 读取 schema-v3。分类分析只使用两种 schema 共有的 logits、labels、
indices，不跨 schema 混用 Fisher。

## 8. 分类统计与判据

### 8.1 Predify timeline

`predify_vs_vanilla_timeline.csv` 固定 11×10=110 行。每行同时保存 Top-1 和
Top-5 的：

- vanilla accuracy；
- no-Hebb Predify accuracy；
- Predify−vanilla 百分点差；
- 配对 bootstrap 95% CI；
- 是否满足“正差值且 CI 下界大于 0”；
- 两个源文件绝对路径。

报告重点是每个条件相对 vanilla 的逐 timestep 差、首次显著超过 vanilla 的
timestep，以及 t1→t9 recurrent refinement。t0 只称为第一个保存的 predictive
state。

### 8.2 Foveated Hebbian interaction

主要指标是：

```text
(Prime − NonPrime)_Hebb
(Prime − NonPrime)_noHebb
[(Prime − NonPrime)_Hebb − (Prime − NonPrime)_noHebb]
```

第三项为差分中的差分。四个 correctness vectors 都按相同 target 对齐后再
bootstrap。

### 8.3 Layout comparison

`foveated_vs_parafovea.csv` 计算：

```text
(Prime − NonPrime)_foveated − (Prime − NonPrime)_parafovea
```

只比较相同 Hebbian model state、相同 odd targets、相同 degradation realization
和相同 `random_init=true` 的结果。

### 8.4 结论用语

只有当百分点差为正且配对 95% CI 不跨 0（实现中 CI 下界严格大于 0）时，才写
“分类性能提升”。否则报告 null 或 negative effect，不用参数调优故事解释。

## 9. 复现命令

### 9.1 单元测试

```powershell
conda run -n predify2021 python -m unittest discover -s tests -v
```

### 9.2 Vanilla 11 点

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments.vanilla_vgg_benchmark `
  --config configs/vanilla_vgg16_degradation_benchmark.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" `
  --device cuda --batch-size 32
```

默认拒绝覆盖已有文件；只有明确重跑时才加 `--overwrite`。

### 9.3 Foveated active Hebbian/no-Hebb

```powershell
conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_foveated_representative.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda

conda run --no-capture-output -n predify2021 `
  python -m predify_experiments `
  --config configs/hebb_vgg_foveated_representative_nohebb.toml `
  --imagenet-root "C:\Users\liang\Documents\ImageNet" --device cuda
```

### 9.4 分类对比与图表

```powershell
conda run -n predify2021 python -m predify_experiments.classification_comparisons `
  --vanilla-dir Hebbian-CNN/Log/vanilla_vgg16_v1 `
  --timeline-predify-dir Hebbian-CNN/Log/foveated_v1/nohebb `
  --timeline-predify-dir Hebbian-CNN/Log/sharpening_v3/nohebb `
  --foveated-hebb-dir Hebbian-CNN/Log/foveated_v1/hebb `
  --foveated-nohebb-dir Hebbian-CNN/Log/foveated_v1/nohebb `
  --parafovea-hebb-dir Hebbian-CNN/Log `
  --parafovea-hebb-dir Hebbian-CNN/Log/sharpening_v3/hebb `
  --output-dir Hebbian-CNN/Log/comparisons_v1 `
  --bootstrap-samples 10000 --seed 114514
```

### 9.5 Foveated representation sharpening

```powershell
conda run -n predify2021 python -m predify_experiments.sharpening_analysis `
  --hebb-dir Hebbian-CNN/Log/foveated_v1/hebb `
  --nohebb-dir Hebbian-CNN/Log/foveated_v1/nohebb `
  --output-csv Hebbian-CNN/Log/comparisons_v1/foveated_sharpening_t9.csv
```

## 10. 完整性校验清单

正式运行完成后必须同时满足：

- vanilla 11/11 文件可反序列化，每个 1000 targets；
- vanilla `trial_indices` 全为 odd，dataset indices 与 Predify odd entries 相等；
- foveated 12/12 文件可反序列化，schema v3、每个 2000 trials；
- `prime_layout`、degradation、condition、Hebbian condition metadata 正确；
- logits shape 为 `[2000,10,1000]`；
- 每层最终 200 个 representation class counts 均为 5；
- normalized Fisher 全为有限值；
- no-Hebb Prime/NonPrime odd logits 逐元素一致；
- 三个 CSV 行数分别为 110、3、3；
- 图与 Markdown 报告均可生成。

## 11. 正式结果

### 11.1 完整性记录

- 34/34 单元测试通过；
- GPU smoke：11 个 vanilla 点与 12 个 foveated 条件全部通过；
- 正式 vanilla 11/11 文件、每个 1000 targets；
- 正式 foveated 12/12 文件、每个 2000 trials、schema v3；
- 每个最终 representation 层均为 200 类×5，normalized Fisher 有限；
- no-Hebb Prime/NonPrime 的全部 odd logits 逐元素完全一致；
- 三个正式 CSV 分别为 110、3、3 行；另有 12 行 foveated sharpening CSV。

### 11.2 Predify 相对 vanilla

t9 Top-1 只有中等噪声条件获得“正差值且配对 CI 下界大于 0”的分类提升：

| 条件 | Predify−vanilla Top-1 pp (95% CI) | 首次显著 t | t1→t9 pp (95% CI) |
|---|---:|---:|---:|
| noise 0.25 | +1.6 [+0.6,+2.7] | 3 | +2.4 [+0.9,+3.9] |
| noise 0.5 | +3.4 [+1.9,+5.0] | 3 | +5.5 [+3.8,+7.3] |
| noise 0.75 | +3.3 [+2.0,+4.7] | 2 | +3.7 [+2.2,+5.2] |
| noise 1.0 | +1.3 [+0.4,+2.3] | 3 | +1.7 [+0.8,+2.7] |

Clean、五个 occlusion points 和 noise 2.0 的 t9 Top-1 CI 均跨 0。尤其 t0
在所有非地板条件都远低于 vanilla，而 t1 出现大幅恢复，进一步证明当前 t0
不能作为 feedforward 基线。分类改善来自整个 recurrent Predify system，不能
单独归因于 feedback。

### 11.3 Foveated priming 分类结果

| 条件 | Hebb P−NP Top-1 pp | difference-in-differences 95% CI |
|---|---:|---:|
| clean | -0.8 | [-1.7,+0.1] |
| noise 0.5 | -1.6 | [-2.7,-0.6] |
| occlusion 0.3 | 0.0 | [-0.9,+0.9] |

三个点都没有显著正向 priming；noise 0.5 是显著负效应。与 parafovea 相比，
noise 0.5 的布局效应差为 -1.4 pp，95% CI [-2.4,-0.4]；clean 与 occlusion
布局差 CI 跨 0。因此完整、清晰、居中的 prime 没有挽救分类 priming，反而在
中等噪声下加重了负效应。

### 11.4 Foveated representation sharpening

使用 activation↓、normalized variance↓、normalized Fisher↑ 的严格三联判据：

- clean：只有 Conv5 完整；
- noise 0.5：Conv4、Conv5 完整；
- occlusion 0.3：只有 Conv5 完整；
- FC1 始终 activation↑，属于 concentration without suppression；
- FC2 的 normalized variance 始终↑，属于 suppression without sharpening。

所以 foveated full prime 仍能在局部 Conv 表征中产生 scale-invariant sharpening，
但没有把它变成分类收益。在 noise 0.5，Conv4/Conv5 的三联 sharpening 与显著
负分类 priming 同时出现，这是“局部几何 sharpening 不等于最终分类改善”的更强
证据。

数值真值位于 `Hebbian-CNN/Log/comparisons_v1/REPORT.md` 和同目录 CSV；报告由
源 `.pckl` 自动生成，避免手工转录误差。本文件保留设计、schema、目录、复现、
完整性与科学解释。
