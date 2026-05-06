# Maestro

Maestro trains a multimodal agent that can call model skills during rollout. The repository includes example train/validation data under `data/` and skill implementations under `skills/`.

The `verl` directory is included from an open-source codebase. Any names, paths, emails, institutional references, examples, or metadata in `verl` originate from the upstream repository and are unrelated to the authors of this project.


## Environment Setup

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


## Start Model Services

Before training, deploy the auxiliary model services. Replace each `/path/to/<model>` placeholder with a local model directory or Hugging Face model id.


Example:

```bash
vllm serve /path/to/Intern-S1-mini --served-model-name Intern-S1-mini --tensor_parallel_size 1 --max-num-seqs 512 --trust-remote-code --port 2368 --gpu_memory_utilization 0.9
```

The default ports used by the skills are:

- `2362`: `qwen3-VL-8B-Instruct`
- `2364`: `Chart-R1`
- `2368`: `Intern-S1-mini`
- `2369`: `medgemma-1.5-4b-it`
- `2370`: `DeepEyes-7B`
- `2376`: `GLM-4.6V-Flash`
- `2388`: `GLM-OCR`
- `2389`: `PR1-Qwen2.5-VL-3B-Detection`

## Training

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
