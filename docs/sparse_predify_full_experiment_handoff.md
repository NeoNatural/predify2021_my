# 稀疏化 Predify 全量实验迁移与执行规范

## 1. 文档目的

本文档用于把当前 `predify2021_my` 工作区已经完成的配置化实验框架，迁移到
“稀疏化predify_my”项目并执行全量实验。它规定：

- 哪些实验方法和代码组件需要移植；
- 稀疏模型必须满足的运行接口；
- 完整实验矩阵、执行顺序与样本划分；
- degradation、prime、PCoder/Hebbian state 的精确定义；
- 结果文件、schema、目录结构与统计口径；
- 稀疏模型新增的必要控制与验收门槛。

当前工作区的数值结果只能作为 dense reference。稀疏模型改变了 representation geometry、
有效维度和 recurrent dynamics，不能直接继承当前 PCoder5 gain、最佳 timestep 或显著性结论。

## 2. 迁移后的核心科学问题

全量实验应把以下命题分开检验：

1. 稀疏 Predify recurrence 是否比 vanilla feedforward 更能抵抗噪声和遮挡；
2. repetition 是否引起 activation suppression；
3. suppression 之外，表征方向是否更集中、类别是否更可分；
4. Same prime 是否使 degraded target 更接近对应 clean representation；
5. 改善来自 Hebbian trace、PCoder state carry，还是两者联合；
6. hidden-state 改善能否被固定下游读出器利用；
7. 稀疏化相对当前 dense reference 改变了哪一个机制环节。

不能用其中一个命题替代另一个。例如：PCoder internal error 下降不等于向 clean target 恢复；
条件内 SVM decoding 提高也不等于固定分类头的准确率提高。

## 3. 不应改变的实验不变量

除非专门进行 ablation，稀疏项目应保持以下条件与当前工作区一致：

| 项目 | 固定设置 |
|---|---|
| 图像预处理 | torchvision `VGG16_Weights.IMAGENET1K_V1` 对应 transform |
| FC 来源 | `fc_weight_source = "torchvision"`；不加载旧 FC 微调 checkpoint |
| recurrent 初始化 | 正式 Predify 条件 `random_init = true` |
| 时间步 | target 10 steps；preview/even trial 2 steps |
| 全局 seed | 114514；degradation base seed 0 |
| Gaussian noise | 在 ImageNet 标准化张量上加 `N(0, σ)` |
| noise realization | 同一图片不同 σ 共用同一标准高斯 realization |
| fractal occlusion | edge-guided fractal black mask，3 octaves |
| 统计 bootstrap | 10,000 次，seed 114514，逐 target 配对 |
| 正式输出 | 每个条件独立文件，包含完整配置快照和逐样本 correctness |

如果稀疏项目不是 VGG16，应保留相同刺激、时间、配对和统计规则，但将 FC 来源字段改成
真实 backbone/head 来源，并在 metadata 中记录稀疏模型 checkpoint hash；不得仍标记为
torchvision VGG16。

## 4. 需要移植的实现组件

### 4.1 建议原样移植

| 当前文件 | 作用 |
|---|---|
| `predify_experiments/config.py` | TOML dataclass、路径解析、多 degradation、validation、CLI override |
| `predify_experiments/sequence.py` | recurrent step、reset、deferred Hebbian commit、PCoder state preserve |
| `Hebbian-CNN/Hebb_Priming_Utils.py` | noise、fractal occlusion、half-field/foveated stimulus |
| `predify_experiments/paper_stimuli.py` | 五大类映射、balanced target、paper parafoveal sequence |
| `predify_experiments/classification_comparisons.py` | vanilla/Predify timeline、foveated/parafovea 比较与报告 |
| `predify_experiments/sharpening_analysis.py` | Hebbian-by-Prime interaction、normalized variance/Fisher |
| `predify_experiments/paper_analysis.py` | identity-grouped SVM、paired bootstrap、recovery/carrier 分析 |
| `predify_experiments/recovery_calibration.py` | PCoder5 selection/held-out calibration |
| `predify_experiments/paper_shared_decoder.py` | 跨图片 frozen five-category readout 与 McNemar 检验 |
| `predify_experiments/paper_figures.py` | clean、recovery、carrier、shared readout 图 |

还应复制 `configs/` 下的对应 TOML 作为模板，以及 `tests/` 中与上述模块对应的测试。

### 4.2 必须按稀疏模型适配

| 当前入口 | 稀疏项目适配点 |
|---|---|
| `predify_experiments/hebb_vgg_priming.py::_build_model` | 构建 sparse backbone，定位 PCoder 与稀疏 hidden layers |
| `predify_experiments/paper_parafoveal.py` | representation hooks、clean reference model、state carrier |
| `predify2021/model_factory/get_model.py` | sparse architecture/checkpoint 加载与 hyperparameter 注入 |
| `Hebbian-CNN/predify_hebb_inject.py` | 将 Hebbian boost 注入正确 PCoder representation 路径 |
| `Hebbian-CNN/Hebbian_VGG_Lib.py` | 如果通道数改变，重建 channel/FC Hebbian layers |
| `Hebbian-CNN/OnlineSharpeningStats_Demo.py` | representation 维度动态化并加入稀疏指标 |

不得硬编码当前 `conv4/conv5=512`、`fc1/fc2=4096`。结果层名可以继续使用
`conv4/conv5/fc1/fc2` 作为跨项目对齐别名，但 metadata 必须同时记录实际模块路径和维度。

### 4.3 模型运行接口契约

实验 runner 假定模型支持：

```text
model(x)       # 新输入的第一个 recurrent step
model(None)    # 后续 recurrent step
model.reset()  # 清空 PCoder recurrent state
model.input_mem = x
```

`pcoder_carry` 条件不会调用会自动 reset 的 `model(x)`，而是写入 `input_mem` 后调用
`model(None)`，保留 preview 的 representation/prediction/gradient memory。

Hebbian layer 至少应暴露：

```text
update_enabled
commit_update()
zero_boost_weight()
```

同一刺激的多个 recurrent steps 期间禁止重复累计 Hebbian 更新。当前实现先将
`update_enabled=False`，最后只执行一次 deferred `commit_update()`。每个新 trial sequence
开始前调用 `zero_boost_weight()`；paper target 阶段不 commit，防止把 target 自身学习误当
成 priming。

## 5. 刺激与退化定义

### 5.1 Clean、noise 与遮挡

- clean 使用 `mode="none", level=0`，不使用 `predify_noise σ=0` 重复保存；
- noise levels：`[0.25, 0.5, 0.75, 1.0, 2.0]`；
- fractal occlusion levels：`[0.1, 0.2, 0.3, 0.4, 0.5]`；
- `predify_noise` 在标准化后的 CHW tensor 上加噪；
- noise seed 对同一 target 固定，使不同 σ 是同一 realization 的幅度缩放；
- fractal seed 必须逐 target 改变。常规 runner 使用 `base_seed + sample_index`；paper runner
  使用 `base_seed + target_offset + ordinal`，保证 selection/held-out 切片和独立运行一致；
- half-field 条件严格保持“先 degradation，再 Gaussian blur，再 downsample”；
- odd target 在所有 prime layout 下必须逐元素相同。

### 5.2 三种 prime layout

1. `parafoveal_half_field`
   - 历史左右拼接刺激；
   - 对比度 0.5、Gaussian blur σ=1，可下采样；
   - Prime 使用随后 odd target；NonPrime 使用 schedule `trial_idx+2` 的另一图像。
2. `foveated_full`
   - even prime 为居中原尺寸 224×224 图像；
   - 不拼接、不降对比、不 blur、不下采样；
   - target、degradation 和 seed 与 half-field 条件配对。
3. `paper_parafoveal`
   - 224×224 灰背景；物体大小为 canvas 的 15%；
   - 中心横向偏移 17.5%；
   - preview 1、preview 2 各含左右两个物体，各运行 2 steps；
   - 中央 target 运行 10 steps；
   - `no_prime/one_back/two_back` 精确控制 target identity 出现位置。

## 6. 全量实验矩阵

所有正式输出建议统一放在：

```text
Hebbian-CNN/Log/sparse_full_v1/
```

不含 PCoder calibration candidates、GPU smoke 和后续多 seed confirmation 时，最低正式
原始结果量为 93 个文件：

| 实验组 | 文件数 | 是否无条件执行 |
|---|---:|---|
| feedforward degradation benchmark | 11 | 是 |
| half-field Hebb/no-Hebb × Prime/NonPrime | 44 | 是 |
| foveated representative | 12 | 是 |
| paper clean benchmark | 6 | 是 |
| recurrent/feedforward recovery | 4 | 是 |
| calibrated memory carriers | 14 | recovery gate 后 |
| shared-readout training inputs | 2 | carrier gate 后 |
| 合计 | 93 | 不含 calibration/confirmation |

当前模板配置的对应关系：

| 实验组 | 配置模板 |
|---|---|
| noise | `hebb_vgg_predify_noise.toml`、`hebb_vgg_predify_noise_nohebb.toml` |
| occlusion | `hebb_vgg_fractal_occlusion.toml`、`hebb_vgg_fractal_occlusion_nohebb.toml` |
| clean | `hebb_vgg_priming.toml`，并为 sparse 项目补一份 matched no-Hebb clean 配置 |
| feedforward | `vanilla_vgg16_degradation_benchmark.toml`，按 sparse backbone 改名/适配 |
| foveated | `hebb_vgg_foveated_representative.toml` 及 `_nohebb` |
| paper clean | `paper_parafoveal_clean_hebb.toml` 及 `_nohebb` |
| recovery | `paper_clean_reference_diagnostic.toml`、`paper_recovery_feedforward_baseline.toml` |
| calibration | `paper_recovery_feedforward_calibration.toml` + `recovery_calibration.py` |
| carrier | `paper_memory_carrier_calibrated.toml` |
| shared readout | `paper_shared_decoder_training.toml` |

### 6.1 Vanilla/稀疏 feedforward degradation benchmark

目的：建立 recurrence 之外的真实 feedforward 基线。

| 条件 | 输入点 | target 数 |
|---|---:|---:|
| clean | 1 | 1000 odd targets |
| noise | 5 | 每点相同 1000 targets |
| occlusion | 5 | 每点相同 1000 targets |

共 11 个文件。若“稀疏化”本身就在 feedforward backbone 中，必须额外运行 sparse
feedforward，而不能直接使用当前 torchvision vanilla 文件。

### 6.2 常规 half-field 全梯度 priming

对 clean、5 个 noise、5 个 occlusion 输入点，运行：

```text
Hebb / no-Hebb × Prime / NonPrime
```

- clean：4 文件；
- noise：5×4=20 文件；
- occlusion：5×4=20 文件；
- 合计 44 文件，每文件沿用 2000 trials，即 1000 odd classification targets。

主要输出：Top-1/Top-5、10-step logits、Hebbian layer metrics、PCoder error、Conv/FC
representation statistics。主要因果量是：

```text
(Prime − NonPrime)_Hebb − (Prime − NonPrime)_noHebb
```

### 6.3 Foveated representative 对比

只跑三个代表点：clean、noise 0.5、occlusion 0.3。

```text
3 degradation points × Hebb/no-Hebb × Prime/NonPrime = 12 files
```

用于判断 full-image foveated priming 是否优于 half-field/parafoveal priming。不要扩展所有
level，除非代表点显示稳定正向 layout interaction。

### 6.4 Paper-matched clean benchmark

```text
no_prime / one_back / two_back × Hebb/no-Hebb = 6 files
```

- 200 targets，五个 supercategories 各 40；
- identity-grouped 5-fold linear SVM；
- 主神经表型要求同一 layer/timestep 同时出现 category decoding 上升和 activation 下降；
- 10 个 timestep 的探索结果不能直接当确认性结果，后续必须预指定时间窗。

### 6.5 Clean-reference recovery

代表点为 noise 0.5、occlusion 0.3：

1. `random_init=true` recurrent target-only：2 文件；
2. 同图、同 degradation realization、`random_init=false` feedforward：2 文件；
3. clean reference model：与 degraded branch 同步 fresh initialization；
4. 记录 PCoder1–5 representation/prediction 到 clean reference 的 cosine、normalized L2、MSE。

严禁再用 recurrent t0 作为 feedforward baseline。`random_init=true` 且 clean/degraded RNG
同步时，t0 会人为接近；正确比较是 recurrent 每个 t 相对 matched `random_init=false`
feedforward state。

### 6.6 稀疏模型专属 PCoder gain calibration

当前 dense 模型选择了 PCoder5：

```toml
ffm = 0.4
fbm = 0.0
erm = 0.0
```

该值不能直接当成稀疏模型最优值，只能作为候选之一。建议复用当前 12-candidate sweep：

- PCoder1–4 固定 `0.8/0.1/0.01`，除非 sparse architecture 已另有验证参数；
- PCoder5 比较 baseline 以及不同 `ffm`、`erm=0/0.01`；
- selection targets 使用 positions 0–99；
- held-out confirmation 使用 positions 100–299；
- ranking 优先最大化 PCoder4/5、两种 degradation 的最差 feedforward-corrected recovery CI；
- 同时要求 FC1/FC2 五类 decoding 不显著低于 feedforward。

如果 sparse model 改变了所有层的动力学，应扩展为逐层 gain calibration，但必须使用嵌套的
selection/confirmation，不能在同一 200 targets 上选参数并报告 CI。

### 6.7 Same/Different × memory carrier 正式实验

校准通过后运行：

| carrier | Same | Different |
|---|---:|---:|
| `none` | 每个 degradation 只保存一次 | 不重复 |
| `hebb_only` | ✓ | ✓ |
| `pcoder_carry` | ✓ | ✓ |
| `hebb_and_pcoder` | ✓ | ✓ |

noise 0.5 与 occlusion 0.3 合计 14 文件，每文件 200 targets。Same/Different 必须拥有完全
相同的 target indices、degradation realization 和 target labels，只有 prime identity 不同。

主要机制判据：

- Same−Different clean-reference recovery；
- Same−Different five-category decoding；
- 同点 activation suppression；
- Top-1/Top-5 只作为更强的工程外推指标。

### 6.8 Shared frozen readout

使用与正式 carrier 测试图片不重叠的 target-only images 训练同一个五分类 head，再固定权重
应用到所有 Same/Different 条件。

迁移到稀疏项目时应改进当前 image-disjoint 设计：先在五个 supercategories 内划分 ImageNet
synsets，使 readout training 与 carrier test 同时做到 image-disjoint 和 synset-disjoint。
保存：

- 每条件、layer、timestep 的 absolute accuracy；
- Same-only correct / Different-only correct；
- paired bootstrap CI；
- exact McNemar 双侧和预声明方向性单侧 p；
- 每个 classifier 的实际迭代数与收敛状态；
- classifier weights、predictions、correctness。

确认性主端点不应重新搜索。dense 结果只提供一个候选：

```text
noise 0.5 + hebb_and_pcoder + FC2 t9 + Same > Different
```

稀疏项目可以把该单元作为跨模型预指定检验，同时把其余 layer/timestep 标为探索性。

## 7. 稀疏模型新增记录

稀疏化会直接降低激活和原始 variance，因此必须在每个 representation layer/timestep 额外保存：

| 指标 | 用途 |
|---|---|
| `zero_fraction` / `active_fraction` | 分离 support sparsity 与幅值变化 |
| `mean_abs_all` | 全单元平均响应，和旧 activation 指标兼容 |
| `mean_abs_active` | 仅非零单元的条件幅值 |
| `active_count_per_sample` | 防止少数样本完全塌缩 |
| `support_jaccard_same_diff` | Same prime 是否稳定激活同一组单元 |
| `normalized_trace_var` | L2 normalization 后的类内离散度 |
| `within_cosine` | 对全局 scaling 不敏感的集中度 |
| `normalized_within_scatter` | 类内几何收缩 |
| `normalized_between_scatter` | 类间结构是否保留 |
| `normalized_fisher_ratio` | between/within separability |
| `participation_ratio` | 有效表征维度，识别过度塌缩 |

完整 repetition sharpening 至少要求：

```text
mean_abs_all ↓
normalized_trace_var ↓（或 within_cosine ↑）
normalized_fisher_ratio ↑
```

对于稀疏模型还应要求 `mean_abs_active`、`active_fraction` 和 participation ratio 没有显示
病理性全局塌缩。只观察到 raw trace variance 下降不能解释为更精确的表征。

## 8. 结果 schema 与命名

### 8.1 常规实验 schema v3

每个 `.pckl` 至少包含：

- `schema_version=3`；
- `run_metadata`：UTC、config path/snapshot、device、seed、condition、layout、degradation、
  Hebbian condition、model/checkpoint identity、稀疏率；
- `idx_log_list`、`full_clss_list`、`y_list`；
- `Out_list`：逐 trial × timestep logits；
- `acc_top_1`、`acc_top_5`、`acc_result`；
- `layer_metrics_arr`；
- `pcoder_error_arr`；
- `repr_stats_timeline` 与最终 `repr_stats`。

文件名必须显式包含 model variant、layout、degradation mode/level、Hebb 状态和
Prime/NonPrime，禁止依赖目录猜测条件。

### 8.2 Paper schema v4 + paper schema v1

除完整配置快照外，必须保存：

- `dataset_indices`、ImageNet labels、supercategory labels、object identity labels；
- 完整 preview/target trial metadata；
- `target_logits [N,10,K]`；
- `target_top1_correct`、`target_top5_correct [N,10]`；
- pooled target representations；
- `representation_recovery [N,10,n_pcoders,3]`；
- `prediction_recovery [N,10,n_pcoders,3]`；
- `pcoder_internal_error [N,10,n_pcoders]`。

稀疏项目中 `K`、层数和 feature dimension 应从模型动态获取并写入 metadata。

## 9. 推荐输出树

```text
Hebbian-CNN/Log/sparse_full_v1/
├── manifest/
│   ├── environment.txt
│   ├── git_commit.txt
│   ├── model_checkpoint_hashes.csv
│   └── run_inventory.csv
├── vanilla_or_sparse_feedforward/
├── half_field/
│   ├── hebb/
│   └── nohebb/
├── foveated/
│   ├── hebb/
│   └── nohebb/
├── comparisons/
├── sharpening/
├── paper/
│   ├── clean/hebb/
│   ├── clean/nohebb/
│   ├── recovery/recurrent/
│   ├── recovery/feedforward/
│   ├── calibration/selection/
│   ├── calibration/confirmation/
│   ├── carriers/
│   ├── shared_decoder/train/
│   └── shared_decoder/analysis/
├── figures/
└── REPORT.md
```

实验大数组继续由 `.gitignore` 排除；配置、分析代码、测试、字段说明和不含敏感路径的小型
CSV 汇总应进入版本控制。

## 10. 分阶段执行顺序

### Stage 0：移植验收

1. 运行所有单元测试；
2. 新增 sparse layer shape、sparsity metrics、state reset/carry 测试；
3. 验证 `σ=0` 输入恒等；
4. 验证不同 σ 共享标准高斯 realization；
5. 验证 fractal seed 跨切片/独立运行一致；
6. 验证 odd target 跨 layout 逐元素一致；
7. 验证 no-Hebb Same/Different 在无 carrier 时一致。

### Stage 1：GPU smoke

- 每个正式条件使用 5–10 targets；
- 检查文件名不冲突、metadata、shape、finite、GPU memory；
- 人工检查一个 noise、一个 occlusion、三个 prime layouts；
- 检查 Hebbian state 每 sequence 只 commit 一次。

### Stage 2：feedforward 与常规全梯度

先跑 11-point feedforward，再跑 44-file half-field matrix 和 12-file foveated representative。
完成后立即生成 timeline、Hebbian interaction 和 scale-invariant sharpening CSV，不等 paper
实验结束才分析。

### Stage 3：paper clean 与 recovery

运行 6-file paper clean、recurrent recovery 和 matched feedforward。只有 deep representation
出现正向 feedforward-corrected recovery，才进入 gain calibration 和 carrier factorial。

### Stage 4：校准与正式 carrier

- selection/held-out split 固定后再运行 sweep；
- 冻结胜出参数；
- 正式运行 14 files；
- 先检查 recovery 和 hidden phenotype，再看 fixed readout/Top-k。

### Stage 5：shared readout 与确认

- 预先构造 synset-disjoint readout/test split；
- 冻结 readout 后分析 Same/Different；
- dense 候选单元作为唯一主检验；
- 若进行 sparse-specific 时空搜索，必须另留一批 targets 作确认。

## 11. 统计判据

1. 所有准确率差使用逐 target 配对，不用两个独立比例近似；
2. 报告效应百分点、95% CI 和原始 absolute accuracy；
3. 二元正确性同时报告 discordant counts 和 exact McNemar；
4. exploratory layer × timestep 图必须标记未校正；
5. “分类提升”要求效应为正且预声明的 95% CI 不跨 0；
6. confirmation 同时要求 Same > Different 和 Same > no-memory/no-prime；
7. sharpening 要求 suppression 与 scale-invariant geometry/separability 同点成立；
8. dense-vs-sparse 比较使用 difference-in-differences，而不是比较两边各自显著与否。

建议的模型交互量：

```text
[(Same − Different)_sparse − (Same − Different)_dense]
```

对 Hebbian 因果效应则使用：

```text
[(Prime − NonPrime)_Hebb − (Prime − NonPrime)_noHebb]
```

## 12. 运行前后审计清单

### 运行前

- [ ] 保存 Git commit、环境、CUDA/PyTorch/torchvision 版本；
- [ ] 保存 sparse checkpoint SHA256、目标稀疏率和实际 layer-wise sparsity；
- [ ] 确认 ImageNet transform 与当前项目一致；
- [ ] 确认 FC/head 来源真实记录；
- [ ] 确认 selection、confirmation、shared-readout target pools 不重叠；
- [ ] 确认正式 config 中没有 portable placeholder path；
- [ ] 确认 output directory 使用 `sparse_full_v1`，不会覆盖 dense 结果。

### 每批运行后

- [ ] 预期文件数全部出现且均可反序列化；
- [ ] 样本数、timestep、类别数和 feature shape 正确；
- [ ] logits、recovery、representation metrics 全部 finite；
- [ ] Same/Different target indices、labels、degradation seeds 完全一致；
- [ ] 每个 level 的 noise/occlusion realization 符合 seed 规则；
- [ ] no-Hebb/no-memory 不出现不应有的 identity effect；
- [ ] SVM/linear head 全部记录 convergence；
- [ ] 输出 CSV 行数与理论矩阵相符；
- [ ] `REPORT.md` 同时记录正、负与 null 结果。

## 13. 命令模板

配置中的 `output_dir` 应改到 `sparse_full_v1` 对应子目录，或使用 CLI override。

```powershell
# 常规 Hebbian / no-Hebbian noise 与 occlusion
python -m predify_experiments.hebb_vgg_priming `
  --config configs/hebb_vgg_predify_noise.toml `
  --imagenet-root "<ImageNet>" --device cuda

python -m predify_experiments.hebb_vgg_priming `
  --config configs/hebb_vgg_predify_noise_nohebb.toml `
  --imagenet-root "<ImageNet>" --device cuda

# Feedforward degradation benchmark
python -m predify_experiments.vanilla_vgg_benchmark `
  --config configs/vanilla_vgg16_degradation_benchmark.toml `
  --imagenet-root "<ImageNet>" --device cuda

# Paper sequence / recovery / carrier 使用同一个 runner
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_parafoveal_clean_hebb.toml `
  --imagenet-root "<ImageNet>" --device cuda

python -m predify_experiments.paper_parafoveal `
  --config configs/paper_memory_carrier_calibrated.toml `
  --imagenet-root "<ImageNet>" --device cuda

# Paper analysis
python -m predify_experiments.paper_analysis `
  --input-dir "<paper result directory>" `
  --output-dir "<paper analysis directory>" `
  --folds 5 --seed 114514 --bootstrap-samples 10000

# Frozen shared readout
python -m predify_experiments.paper_shared_decoder `
  --train-dir "<shared readout training directory>" `
  --evaluation-dir "<carrier result directory>" `
  --output-dir "<shared readout analysis directory>" `
  --seed 114514 --bootstrap-samples 10000 --max-iter 50000
```

在稀疏 runner 尚未保持当前 module name 时，可以保留命令接口但替换内部 `_build_model`；
不要为了少改代码而让 metadata 继续写成 dense VGG。

## 14. 当前 dense reference 的迁移提示

当前 dense 工作区已确认：

- corrected recurrence 能改善 deep clean-reference recovery；
- PCoder carry 较早但衰减，Hebbian recovery 较慢但持续；
- Same prime 在隐藏层可产生时间局限的 suppression + category decoding enhancement；
- torchvision 原始 1000-way head 在 15% paper objects 上处于 floor；
- frozen five-category head 在 noise 0.5、联合 carrier、FC2 t9 出现 +2.5 pp 方向性候选，
  但 exact McNemar 双侧 p=0.0625，尚未确认。

稀疏项目的价值不在于“复制显著点”，而在于判断稀疏化是否：

1. 提高 noisy evidence 的 recurrent integration；
2. 让 Hebbian trace 更选择性，而非仅增加零激活；
3. 保持或提高 normalized Fisher separability；
4. 使 hidden recovery 更容易落到固定读出器的有利决策方向；
5. 对 occlusion 中真正缺失的信息产生比 dense decoder 更强的补全。

最终报告应以完整时间线和预声明端点为主，不以事后最显著的 layer/timestep 作为项目成败。
