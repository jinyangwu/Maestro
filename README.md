<h1 align="center">
MAESTRO: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles
</h1>

<div align="center">
<p>
    <a href="https://arxiv.org/abs/2605.22177">
      <img src="https://img.shields.io/badge/Paper-arxiv%3A2605.22177-blue" alt="Paper"/>
    </a>
    <a href="https://huggingface.co/papers/2605.22177">
      <img src="https://img.shields.io/badge/Daily%20Paper-huggingface-yellow" alt="HF Paper"/>
    </a>
  </p>
</div>

## 🔥 Overview

**MAESTRO** trains a lightweight multimodal orchestrator that dynamically calls expert models and task-specific skills during rollout. Instead of relying on a single monolithic model or fixed tool-selection logic, MAESTRO learns to coordinate a hierarchical model-skill ensemble through reinforcement learning.

The repository includes example train/validation data under `data/` and skill implementations under `skills/`.

> **Note**
> The `verl` directory is included from an open-source codebase. Any names, paths, emails, institutional references, examples, or metadata in `verl` originate from the upstream repository and are unrelated to the authors of this project.

## 🗞️ News

- **`2026-05-21`**: We released the MAESTRO repository with example data, skills, and training scripts.

## 🛠️ Installation

### Python environment

Create a Python environment and install the dependencies from the repository root:

```bash
conda create -n maestro python=3.10 -y
conda activate maestro
pip install -r requirements.txt
```

Set an OpenAI API key before training:

```bash
export OPENAI_API_KEY=<your_api_key>
```

## 🚀 Start Model Services

Before training, deploy the auxiliary model services. Replace each `/path/to/<model>` placeholder with a local model directory or Hugging Face model id.

Example:

```bash
vllm serve /path/to/Intern-S1-mini --served-model-name Intern-S1-mini --tensor_parallel_size 1 --max-num-seqs 512 --trust-remote-code --port 2368 --gpu_memory_utilization 0.9
```

The default ports used by the skills are:

| Port | Model service |
| --- | --- |
| `2362` | `qwen3-VL-8B-Instruct` |
| `2364` | `Chart-R1` |
| `2368` | `Intern-S1-mini` |
| `2369` | `medgemma-1.5-4b-it` |
| `2370` | `DeepEyes-7B` |
| `2376` | `GLM-4.6V-Flash` |
| `2388` | `GLM-OCR` |
| `2389` | `PR1-Qwen2.5-VL-3B-Detection` |

## 🧠 Training

The default training script uses:

- Training data: `data/train_data_example.parquet`
- Validation data: `data/val_data_example.parquet`
- Images referenced through relative paths under `data/images/`

Start training with:

```bash
bash train.sh
```

To train from a local checkpoint or a different model id, override `MODEL_NAME`:

```bash
MODEL_NAME=/path/to/Qwen3-VL-4B-Thinking bash train.sh
```

## 📁 Repository Structure

```text
Maestro/
├── data/          # Example train/validation data and image assets
├── skills/        # Model-skill implementations used during rollout
├── verl/          # Upstream training framework code
├── train.sh       # Default MAESTRO training entrypoint
└── requirements.txt
```

## ⭐ Citation

If you find this project useful, welcome to cite us.

```bibtex
@misc{wu2026maestro,
      title={MAESTRO: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles},
      author={Jinyang Wu and Guocheng Zhai and Ruihan Jin and Yuhao Shen and Zhengxi Lu and Fan Zhang and Haoran Luo and Zheng Lian and Zhengqi Wen and Jianhua Tao},
      year={2026},
      eprint={2605.22177},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.22177}, 
}
```

## 🤝 Acknowledgement

This project builds on the open-source `verl` ecosystem and uses external expert models and model-serving infrastructure such as vLLM. We thank the authors and contributors of these projects.
