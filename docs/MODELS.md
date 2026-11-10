# Released Memory Adapters

mini-engram releases two public, ungated LoRA adapters for **Qwen/Qwen2.5-7B-Instruct**. Both store knowledge from programmatically generated fictional company wikis. Their Hugging Face repositories include `adapter_config.json` and `adapter_model.safetensors`; the base model must be loaded separately.

## Model inventory

Verified against the public Hugging Face repository API and adapter configurations on **2026-11-10**:

| Adapter | Training corpus | Published base → adapter token-F1 | Snapshot revision |
| :--- | :--- | :--- | :--- |
| [tersemind/mini-engram-xinglan](https://huggingface.co/tersemind/mini-engram-xinglan) | Xinglan Tech wiki; 123 self-study QA pairs | 0.126 → 0.666 | `fcfa49ad5d111b3eb29430e416e27862a9971424` |
| [tersemind/mini-engram-hanhai](https://huggingface.co/tersemind/mini-engram-hanhai) | Hanhai Robotics wiki; 93 self-study QA pairs | 0.113 → 0.664 | `ce44ce00c565f31d654d1fea5746140c4665cd5d` |

Both model cards describe **rank 16, 40,370,176 adapter parameters, bf16, approximately 81 MB**. Their configurations specify LoRA alpha 32 and all seven attention/MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

The published config retains a training-machine path in `base_model_name_or_path`. Load the public base ID explicitly as shown below; that old `/root/...` path is not a prerequisite.

The F1 numbers are the model cards' reported demo results. This homepage update verified public availability and configuration, without rerunning GPU inference or evaluation. The demo snapshots do not replace the tenant-specific adapters used for LoCoMo, LongHealth or the LongMemEval-S extension.

## Load one adapter with PEFT

On a CUDA machine with enough GPU memory, install the inference dependencies in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers peft accelerate huggingface_hub
```

Then load the base and attach one adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id = "tersemind/mini-engram-xinglan"
adapter_revision = "fcfa49ad5d111b3eb29430e416e27862a9971424"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id, revision=adapter_revision)
model.eval()

messages = [
    {"role": "system", "content": "你是星澜科技内部知识助手，请根据已掌握的公司知识简洁准确地回答问题。"},
    {"role": "user", "content": "星澜科技的报销系统叫什么？"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For Hanhai, select the second repository and revision from the table, use the Hanhai company system prompt, and ask a question from its wiki. The pinned revision fixes the adapter snapshot; a fully frozen reproduction should also pin the base-model revision and dependency versions.

## Serve both tenants from this repository

Set up the project's GPU environment as described in the [Quick Start](../README.md#quick-start), then download the adapter files to the directories discovered by `scripts/serve.sh`:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

for tenant, revision in [
    ("xinglan", "fcfa49ad5d111b3eb29430e416e27862a9971424"),
    ("hanhai", "ce44ce00c565f31d654d1fea5746140c4665cd5d"),
]:
    snapshot_download(
        repo_id=f"tersemind/mini-engram-{tenant}",
        revision=revision,
        local_dir=f"data/adapters/{tenant}",
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
PY

bash scripts/serve.sh
```

In a second terminal:

```bash
.venv/bin/python scripts/chat.py --tenant xinglan "星澜科技的报销系统叫什么？"
.venv/bin/python scripts/chat.py --tenant hanhai "瀚海机器人最大的客户是谁？"
```

The chat script queries both the base model and the requested tenant adapter. `scripts/serve.sh` resolves its base through ModelScope by default and exposes a vLLM OpenAI-compatible endpoint on port 8000. The project uses the OpenAI client library to talk to this local endpoint; these calls do not invoke a hosted OpenAI model.

## Use and redistribution

Code and project-owned adapter contributions are MIT-licensed. The adapters derive from **Qwen2.5-7B-Instruct (Apache 2.0)**; redistribution must retain the base-model license and notices. See each [HF model card](https://huggingface.co/tersemind/mini-engram-xinglan#license), [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md).

These are research demos on fictional company knowledge. The study observed confident answers to out-of-scope questions, so tenant routing and refusal behavior need separate validation. Consult the [benchmark notes](BENCHMARKS.md) for the measured limitations and the [data notes](../data/README.md) for benchmark-specific terms.
