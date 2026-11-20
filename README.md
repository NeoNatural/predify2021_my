# predify2021

Code for reproducing the results presented in the paper 'Predify:Augmenting deep neural networks with brain-inspired predictive coding dynamics' (https://arxiv.org/pdf/2106.02749.pdf)


## Dependencies
<pre>
- predify  (<a href="https://github.com/miladmozafari/predify">Repository</a>)
- torch 
- tensorboard
- loguru

For EfficientNetB0:
- timm     (<a href="https://github.com/rwightman/pytorch-image-models">Repository</a>)

For Adversarial Attacks:
- foolbox 3.x
</pre>


## Repository structure
-  `model_factory` provides predified versions of the models (VGG and EfficientNetB0) using `get_model` function. 
-  `adversarial_attacks` contains all scripts for performing and analysing adversarial attacks performed in the paper.
-  `mCE_scores` contains scripts for performing and calculating mCE scores on the predified networks.
-  `manifold_projection` contains scripts for calculating the correlation distances between clean and noisy representations (refer paper for more details).

## Weights of all the models

[Link to the PEfficientNetB0 weights](https://www.dropbox.com/s/0np1pzp3o3qhonv/weights_pefficientNetB0_imagenet.zip?dl=0) 

[Link to the PVGG_Weights](https://www.dropbox.com/s/8lzp6wfo6n3bymk/weights_pvgg16_imagenet.zip?dl=0)


## Reproducible experiment entry point

The actively maintained experiment infrastructure lives in
`predify_experiments`. The original research modules remain under
`predify2021`, while the Hebbian scripts in `Hebbian-CNN` are retained as
compatibility entry points.

Create a clean environment instead of reusing Anaconda `base`:

```powershell
conda env create -f environment.yml
conda activate predify2021
python -m pip install -e .
python -m unittest discover -s tests -v
```

Resolve and validate an experiment without loading PyTorch or ImageNet:

```powershell
$env:PREDIFY_IMAGENET_ROOT = "E:\ImageNet"
python -m predify_experiments --check-config
```

Run the current priming and non-priming experiment:

```powershell
python -m predify_experiments
```

Common settings can be changed in `configs/hebb_vgg_priming.toml` or
overridden without editing code:

```powershell
python -m predify_experiments `
  --imagenet-root "E:\ImageNet" `
  --output-dir "E:\predify-results" `
  --device cuda `
  --max-samples 20
```

### Input degradation and prime blur

The stimulus pipeline supports three mutually exclusive input-degradation
modes under `[stimulus]`:

- `degradation_mode = "none"` keeps the original experiment.
- `degradation_mode = "predify_noise"` applies additive Gaussian noise to the
  ImageNet-normalized tensor. If `degradation_levels` is omitted, the exact
  Predify sigma values `0.25, 0.5, 0.75, 1.0, 2.0` are run.
- `degradation_mode = "fractal_occlusion"` applies edge-guided fractal black
  occlusion without additive noise. If `degradation_levels` is omitted, 30
  intensities from `0.0` through `0.5` are run.

`blur_sigma` remains the independent Gaussian blur used by the dual-image
prime stimulus. For a prime canvas, processing is ordered as input degradation,
then prime Gaussian blur, then the existing optional downsampling. Each mode
and level is written to a distinct result file.

The dedicated torchvision-FC Predify-noise experiment can be launched with:

```powershell
python Hebbian-CNN/Hebb_VGG_Pred_Run.py `
  --imagenet-root "E:\ImageNet" `
  --device cuda
```

Its configuration is `configs/hebb_vgg_predify_noise.toml`; it runs sigma
values `0.25, 0.5, 0.75, 1.0, 2.0` for both Prime and NonPrime while
retaining the torchvision VGG16 classifier weights and active Hebbian layers.

The companion `configs/hebb_vgg_fractal_occlusion.toml` runs fractal
occlusion intensities `0.1, 0.2, 0.3, 0.4, 0.5` under the same model and
condition settings. The level-zero identity condition is not rerun because
the earlier clean-input experiment already supplies that baseline.

### Scale-invariant repetition-sharpening analysis

The complete Chinese experiment protocol, schema-v3 field reference, result
directory layout, validation record, and interpretation are documented in
`docs/vgg16_predify_noise_occlusion_sharpening_v3.md`.

Result schema v3 supplements the raw within-class `trace_var` with
`normalized_trace_var`, computed after L2-normalizing each representation.
It also records normalized within- and between-class scatter and their Fisher
ratio. A strict complete-sharpening result therefore requires activation to
decrease, normalized within-class variance to decrease, and normalized Fisher
separability to increase.

The matched no-Hebbian configurations keep the torchvision VGG16 classifier,
Predify PCoders, degradation, seeds, and trial schedule unchanged while using
`original_mode = true` to pass through the four Hebbian modules:

- `configs/hebb_vgg_predify_noise_nohebb.toml`
- `configs/hebb_vgg_fractal_occlusion_nohebb.toml`

After running the active and no-Hebbian conditions into separate directories,
calculate the Hebbian-by-Prime interaction with:

```powershell
python -m predify_experiments.sharpening_analysis `
  --hebb-dir Hebbian-CNN/Log/sharpening_v3/hebb `
  --nohebb-dir Hebbian-CNN/Log/sharpening_v3/nohebb `
  --output-csv Hebbian-CNN/Log/sharpening_v3/hebbian_interaction_t9.csv
```

The CSV contains matched Top-1/Top-5 accuracy, the three sharpening effects,
their difference-in-differences, the strict interaction criterion, and the
four source result paths for every degradation level and representation layer.

### Vanilla timeline and foveated full-image priming

The follow-up protocol adds a true torchvision VGG16 feedforward benchmark and
a centered full-image prime. Current `random_init=true` Predify t0 is treated as
the first saved predictive state, not as vanilla feedforward. Full design,
schemas, file tree, reproducibility commands, paired-bootstrap decision rules,
and validation requirements are in
`docs/vanilla_predify_timeline_foveated_v1.md`.

The three dedicated configurations are:

- `configs/vanilla_vgg16_degradation_benchmark.toml`
- `configs/hebb_vgg_foveated_representative.toml`
- `configs/hebb_vgg_foveated_representative_nohebb.toml`

The new stimulus matrix format can declare multiple degradation modes in one
configuration using `[[stimulus.degradations]]`; legacy single-mode TOML files
remain compatible. Foveated filenames include `_foveated_full`, while legacy
half-field filenames remain unchanged.

The cross-experiment synthesis, computational/neuroscience interpretation, and
the proposed clean-prime/state-carry follow-up are documented in
`docs/results_synthesis_mechanism_and_next_experiments.md`.

### Paper-matched parafoveal priming

For porting the complete experiment framework to the sparse Predify project,
including the full run matrix, model API contract, sparse-specific metrics,
result schemas, directory layout, gates, and audit checklist, see
`docs/sparse_predify_full_experiment_handoff.md`.

The paper-matched no/1-back/2-back protocol, five-category identity-grouped
decoding, clean-reference PCoder diagnostics, memory-carrier implementation,
schema-v4 field reference, result tree, formal results, and go/no-go decision
are documented in `docs/paper_parafoveal_v1_execution.md`.

```powershell
python -m predify_experiments.paper_parafoveal `
  --config configs/paper_parafoveal_clean_hebb.toml `
  --imagenet-root "E:\ImageNet" --device cuda

python -m predify_experiments.paper_analysis `
  --input-dir Hebbian-CNN/Log/paper_parafoveal_v1/representations/hebb `
  --output-dir Hebbian-CNN/Log/paper_parafoveal_v1/decoding/hebb
```

The first 200-target run found a Hebbian-dependent one-back Conv4 decoding
head-start at target t0, but no layer/timestep passed the strict joint
suppression-plus-decoding criterion. Recovery must be compared with a matched
`random_init=false` feedforward reference: recurrent t0 is artificially close
between clean and degraded branches when their random initialization is
synchronized. With the corrected baseline, deep recurrent recovery is positive
and a PCoder5-only calibration selected `ffm=0.4, fbm=0, erm=0` on targets
disjoint from confirmation. The completed 14-file carrier factorial shows
early paper-like suppression plus category decoding and persistent object-specific
clean recovery, while the original torchvision 1000-way head remains at floor.
A shared frozen five-category readout trained on separate images yields one
directional terminal candidate at noise 0.5, combined Hebbian+PCoder carry,
FC2 t9: +2.5 pp with bootstrap 95% CI [+0.5,+5.0]. Its exact paired two-sided
McNemar p-value is 0.0625 and the training/test synsets are not fully disjoint,
so it is a held-out-confirmation hypothesis rather than a confirmed terminal
benefit. See the execution document for the complete audit and next protocol.

The legacy commands `python Hebbian-CNN/Hebb_VGG_Pred_Run.py` and
`python Hebbian-CNN/Hebb_VGG_Pred_Run_2.py` dispatch to the same runner. The
first selects the dedicated torchvision-FC Predify-noise experiment; the
second selects the existing fine-tuned FC checkpoint with the general priming
configuration.


## Research snapshot

This branch includes experiment code, configs, tests, documentation, and a
selection of aggregate results. Full experiment archives, saved activations,
checkpoints, datasets, and generated figures remain in the research workspace.
Historical reports may reference these local artifacts. Re-running analyses
that read caches requires the original data or a regenerated cache.

- [Training-set expansion and augmentation](docs/priming_sharpen_20261112/expansion_20261118/summary.csv)
- [FC2 early gate calibration and evaluation](docs/hebbian_week_20261112_14/timecourse_20261119/fc2_ablation/lateral/early_gate/index.html)
- [FC2 middle and late replication](docs/hebbian_week_20261112_14/timecourse_20261119/fc2_ablation/lateral/fresh_replication/index.html)

A clean clone needs local PCoder checkpoints before the configuration check
can pass. The tripartite manifest tests also require the archived ImageNet
and THINGS manifests under Hebbian-CNN/Log/tripartite_20261027.
