# Latent-CDM

Latent-CDM is a research codebase for dynamic latent communication in sequential clinical diagnosis.

The first milestone is a diagnosis-only adapter:

- input: partial patient state from MIMIC-CDM;
- output: exactly one disease name immediately after the prompt;
- no verbal confidence or explanation.

This keeps the output space closed and makes the model suitable for later hidden-state, disease-log-likelihood, and dynamic decoding analyses.

The planned architecture uses one shared LLM backbone with two LoRA adapters:

- diagnosis adapter: trained first, then frozen as a diagnostic belief encoder;
- planner adapter: trained later for evidence-acquisition actions.

After the diagnosis adapter is trained, a lightweight stop/add classifier will be attached to its final hidden state. If the classifier predicts stop, the diagnosis adapter decodes the disease label. If it predicts add, the latent diagnostic state can be passed to the planner adapter through hidden-state or KV-cache handoff.

## Layout

```text
Latent-CDM/
├── configs/              # experiment configs
├── dataset/              # local datasets, ignored by git
├── prompts/              # prompt templates
├── scripts/              # analysis/evaluation entry points
├── src/
│   ├── data/             # dataset-specific state rendering
│   ├── models/           # model/tokenizer/LoRA loading
│   └── training/         # adapter/gate training loops
└── train.py              # main entry point
```

## Data

By default, configs point to:

```text
dataset/
├── train.csv
├── val.csv
├── test.csv
└── lab_test_mapping.csv
```

Dataset rendering is selected by `data.builder`. The current builder is `mimic_cdm`; additional datasets should add a builder under `src/data/` and register it in `build_state_builder`.

## Diagnosis Adapter Training

```bash
cd /home/seojun/Workspace/AAAI/LatentCDM
export HF_HOME=/media/NAS/nas_175/seojun/LatentCDM/cache/huggingface
export TORCH_HOME=/media/NAS/nas_175/seojun/LatentCDM/cache/torch
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/diagnosis_adapter.yaml
```

Runs are saved under `training.run_root` with experiment, dataset, date, and time.
The default points to NAS storage so large checkpoints and logs do not fill the local disk:

```text
/media/NAS/nas_175/seojun/LatentCDM/runs/{experiment_name}/{data.name}/YYYY-MM-DD/HH-MM-SS/
```

For a small sanity run:

```bash
export HF_HOME=/media/NAS/nas_175/seojun/LatentCDM/cache/huggingface
export TORCH_HOME=/media/NAS/nas_175/seojun/LatentCDM/cache/torch
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/smoke.yaml
```
