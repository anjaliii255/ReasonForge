import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
NB_DIR = ROOT / "notebooks"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def writefile(name):
    return code(f"%%writefile {name}\n" + (ANALYSIS / name).read_text())


def has_outputs(path):
    if not path.exists():
        return False
    return any(c.get("outputs") for c in json.loads(path.read_text())["cells"])


def save(name, cells):
    path = NB_DIR / name
    if has_outputs(path) and "--force" not in sys.argv:
        print("skip", path, "(has run outputs; use --force to overwrite)")
        return
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", NB_DIR / name)


INSTALL = code(r'''
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.11.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq {_triton} "huggingface_hub>=0.34.0" "datasets==4.3.0"
try:
    import importlib.metadata as _md; _torch_v = tuple(int(_p) for _p in _md.version("torch").split("+")[0].split(".")[:2])
except Exception:
    _torch_v = ()
_torchao = "torchao>=0.16.0" if _torch_v >= (2, 10) else "torchao>=0.16.0,<0.18.0"
!uv pip install -qqq --no-deps --upgrade "{_torchao}"
!uv pip install -qqq transformers==4.56.2
!uv pip install -qqq --no-deps trl==0.22.2
!uv pip install -qqq math-verify
print("is_t4:", is_t4, "| vllm:", _vllm)
''')

ENV = code(r'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"; os.environ["USE_TF"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import importlib.metadata as md
for pkg in ["torch", "vllm", "transformers", "trl", "peft", "unsloth", "datasets", "math-verify"]:
    try: print(f"{pkg:13s} {md.version(pkg)}")
    except md.PackageNotFoundError: print(f"{pkg:13s} NOT INSTALLED")

import torch
print("GPU:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
''')


def build_05():
    cells = [
        md("# 05 - Fast evaluation with vLLM (base + every adapter found)"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config + adapter discovery"),
        code(r'''
import glob, os, shutil

MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS  = ["gsm8k", "math500"]
STYLES    = ["boxed", "plain"]
MAX_NEW   = 4096
N_LIMIT   = None
OUT_DIR   = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)

for p in glob.glob("/kaggle/input/**/results/*.pkl", recursive=True):
    dst = os.path.join(OUT_DIR, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst); print("resuming from", p)

def short_name(path):
    s = path.lower()
    if "grpo" in s:
        return "grpo_b" if ("grpo_b" in s or "grpo-b" in s) else "grpo_a"
    if "sft" in s:
        return "sft"
    return os.path.basename(path.rstrip("/"))

ADAPTERS = {}
for cfg in sorted(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)):
    d = os.path.dirname(cfg)
    if "checkpoint-" in d:
        continue
    ADAPTERS.setdefault(short_name(d), d)
print("adapters:", ADAPTERS if ADAPTERS else "NONE (base only) — did you Add Data?")
'''),
        md("## 5. Load vLLM"),
        code(r'''
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=MODEL_ID,
    dtype="half",
    max_model_len=MAX_NEW + 1024,
    gpu_memory_utilization=0.85,
    enable_lora=bool(ADAPTERS),
    max_lora_rank=64,
    max_loras=1,
    seed=0,
)
tokenizer = llm.get_tokenizer()

GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, repetition_penalty=1.0)
SAMPLED = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=MAX_NEW, seed=0)
SAMPLED_FOR = ["base", "sft"]

RUNS = {"base": (None, GREEDY, STYLES)}
for i, (name, path) in enumerate(ADAPTERS.items(), start=1):
    RUNS[name] = (LoRARequest(name, i, path), GREEDY, STYLES)
for name in SAMPLED_FOR:
    if name in RUNS:
        RUNS[f"{name}_t06"] = (RUNS[name][0], SAMPLED, ["boxed"])
print("runs:", {k: v[2] for k, v in RUNS.items()})
'''),
        md("## 6. Generate"),
        code(r'''
import time
from rf_eval import load_eval_set, chat_prompts, rows_from_outputs, quick_acc, load_result, save_result

summary = []
for ds_name in DATASETS:
    items = load_eval_set(ds_name, N_LIMIT)
    for style in STYLES:
        path = f"{OUT_DIR}/{ds_name}_{style}.pkl"
        result = load_result(path, {"dataset": ds_name, "style": style, "n": len(items),
                                    "max_new": MAX_NEW, "model_id": MODEL_ID})
        prompts = chat_prompts(tokenizer, items, style)
        for name, (lora, sp, styles) in RUNS.items():
            if style not in styles:
                continue
            if len(result.get(name, [])) == len(items):
                print(f"[skip] {ds_name}/{style}/{name} already done")
            else:
                t0 = time.time()
                outs = llm.generate(prompts, sp, lora_request=lora)
                result[name] = rows_from_outputs(items, outs)
                result["config"].setdefault("decoding", {})[name] = f"temperature={sp.temperature}, top_p={sp.top_p}"
                save_result(path, result)
                print(f"[done] {ds_name}/{style}/{name}: {(time.time()-t0)/60:.1f} min")
            rows = result[name]
            summary.append((ds_name, style, name, quick_acc(rows),
                            sum(r["truncated"] for r in rows) / len(rows),
                            sum(r["has_box"] for r in rows) / len(rows)))

print(f"\n{'dataset':8s} {'style':6s} {'model':8s} {'acc':>6s} {'trunc':>6s} {'boxed':>6s}")
for d, s, m, a, t, b in summary:
    print(f"{d:8s} {s:6s} {m:8s} {a*100:5.1f}% {t*100:5.1f}% {b*100:5.1f}%")
'''),
        md("## 7. Package for download"),
        code(r'''
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", "results")
print("download /kaggle/working/results.zip from the Output panel")
'''),
    ]
    save("05-eval-vllm.ipynb", cells)


def build_06():
    cells = [
        md("# 06 - GRPO on GSM8K from base Qwen2.5-1.5B-Instruct (Option A)"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config"),
        code(r'''
SMOKE_TEST = True

RUN_NAME          = "grpo_a"
MODEL_ID          = "unsloth/Qwen2.5-1.5B-Instruct"
LOAD_4BIT         = False
LORA_R, LORA_ALPHA = 32, 64

NUM_GENERATIONS   = 8
PROMPTS_PER_STEP  = 4
MAX_PROMPT_LEN    = 320
MAX_COMPLETION    = 1024
LR                = 1e-5
BETA              = 0.0
NUM_ITERATIONS    = 1
EPSILON, EPSILON_HIGH = 0.2, 0.28
FORMAT_WEIGHT     = 0.1
MAX_STEPS         = 600
SAVE_STEPS        = 25
TIME_BUDGET_HOURS = 10.0

EVAL_SETS         = ["gsm8k", "math500"]
EVAL_N            = None

if SMOKE_TEST:
    MAX_STEPS, SAVE_STEPS, EVAL_N, TIME_BUDGET_HOURS = 5, 5, 50, 1.0

OUT_DIR   = f"/kaggle/working/{RUN_NAME}"
FINAL_DIR = f"/kaggle/working/{RUN_NAME}-final"
print(f"SMOKE_TEST={SMOKE_TEST}  MAX_STEPS={MAX_STEPS}  completions/step={NUM_GENERATIONS*PROMPTS_PER_STEP}")
'''),
        md("## 5. Weights & Biases"),
        code(r'''
REPORT_TO = "none"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    os.environ["WANDB_PROJECT"] = "reasonforge"
    REPORT_TO = "wandb"
except Exception as e:
    print("W&B disabled (no WANDB_API_KEY secret):", type(e).__name__)
print("report_to:", REPORT_TO)
'''),
        md("## 6. Load model"),
        code(r'''
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_PROMPT_LEN + MAX_COMPLETION + 64,
    load_in_4bit=LOAD_4BIT,
    fast_inference=True,
    max_lora_rank=LORA_R,
    gpu_memory_utilization=0.85,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(f"allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''),
        md("## 7. Data + rewards"),
        code(r'''
from datasets import load_dataset
from rf_extract import user_prompt, gsm8k_gold, extract_boxed, answers_match, has_boxed

train = load_dataset("openai/gsm8k", "main", split="train")
train = train.map(lambda r: {"prompt": [{"role": "user", "content": user_prompt(r["question"], "boxed")}],
                             "gold": gsm8k_gold(r["answer"])},
                  remove_columns=train.column_names)
print(train)
print(train[0]["prompt"][0]["content"], "\n-> gold:", train[0]["gold"])

_calls = {"n": 0}

def correctness_reward(completions, gold, **kwargs):
    texts = [c[0]["content"] for c in completions]
    rewards = [1.0 if answers_match(extract_boxed(t), g) else 0.0 for t, g in zip(texts, gold)]
    _calls["n"] += 1
    if _calls["n"] % 25 == 1:
        print(f"\n--- sample (call {_calls['n']}) gold={gold[0]} pred={extract_boxed(texts[0])} r={rewards[0]}\n"
              f"{texts[0][-600:]}\n---")
    return rewards

def format_reward(completions, **kwargs):
    return [FORMAT_WEIGHT if has_boxed(c[0]["content"]) else 0.0 for c in completions]
'''),
        md("## 8. Trainer"),
        code(r'''
import glob, json, re, time
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

def latest_checkpoint():
    cands = glob.glob("/kaggle/input/**/checkpoint-*/trainer_state.json", recursive=True)
    cands += glob.glob(f"{OUT_DIR}/checkpoint-*/trainer_state.json")
    cands = [os.path.dirname(c) for c in cands if RUN_NAME in c]
    return max(cands, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)), default=None)

RESUME = None if SMOKE_TEST else latest_checkpoint()
print("resume from:", RESUME)

class TimeBudgetAndLogs(TrainerCallback):
    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600
        self.t_last = None
    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.t_last is not None and state.global_step % 5 == 0:
            print(f"step {state.global_step}: {now - self.t_last:.0f}s/step")
        self.t_last = now
        if now > self.deadline:
            print(f"time budget reached at step {state.global_step} -> saving and stopping")
            control.should_save = True
            control.should_training_stop = True
    def on_save(self, args, state, control, **kw):
        with open(f"{args.output_dir}/log_history.json", "w") as f:
            json.dump(state.log_history, f)

args = GRPOConfig(
    output_dir=OUT_DIR,
    run_name=RUN_NAME,
    report_to=REPORT_TO,
    seed=42,
    learning_rate=LR,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    optim="adamw_8bit",
    adam_beta1=0.9, adam_beta2=0.99,
    weight_decay=0.0,
    max_grad_norm=0.2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    per_device_train_batch_size=NUM_GENERATIONS,
    gradient_accumulation_steps=PROMPTS_PER_STEP,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION,
    temperature=1.0,
    use_vllm=True,
    mask_truncated_completions=True,
    loss_type="dapo",
    beta=BETA,
    num_iterations=NUM_ITERATIONS,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    max_steps=MAX_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=args,
    train_dataset=train,
    callbacks=[TimeBudgetAndLogs(TIME_BUDGET_HOURS)],
)
'''),
        md("## 9. Train"),
        code(r'''
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
print(f"training wall time: {(time.time()-t0)/3600:.2f} h")
'''),
        md("## 10. Save adapter + logs"),
        code(r'''
import shutil
model.save_lora(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
with open(f"{FINAL_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)

shutil.make_archive(f"/kaggle/working/{RUN_NAME}_adapter", "zip", FINAL_DIR)
os.makedirs("/kaggle/working/logs", exist_ok=True)
shutil.copy(f"{FINAL_DIR}/log_history.json", f"/kaggle/working/logs/{RUN_NAME}_log_history.json")
shutil.make_archive(f"/kaggle/working/{RUN_NAME}_logs", "zip", "/kaggle/working/logs")
print(os.listdir(FINAL_DIR))
'''),
        md("## 11. Eval: base vs GRPO"),
        code(r'''
from vllm import SamplingParams
from rf_eval import load_eval_set, chat_prompts, rows_from_outputs, quick_acc, load_result, save_result

RES_DIR = "/kaggle/working/results"
os.makedirs(RES_DIR, exist_ok=True)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_COMPLETION, repetition_penalty=1.0)
lora = model.load_lora(FINAL_DIR)

for ds_name in EVAL_SETS:
    items = load_eval_set(ds_name, EVAL_N)
    prompts = chat_prompts(tokenizer, items, "boxed")
    path = f"{RES_DIR}/{ds_name}_boxed_{RUN_NAME}.pkl"
    result = load_result(path, {"dataset": ds_name, "style": "boxed", "n": len(items),
                                "max_new": MAX_COMPLETION, "engine": "unsloth-vllm", "decoding": "greedy"})
    for name, lr in [("base", None), (RUN_NAME, lora)]:
        t1 = time.time()
        outs = model.fast_generate(prompts, sampling_params=sp, lora_request=lr)
        result[name] = rows_from_outputs(items, outs)
        save_result(path, result)
        rows = result[name]
        print(f"{ds_name:8s} {name:8s} acc {quick_acc(rows)*100:5.1f}%  "
              f"trunc {sum(r['truncated'] for r in rows)/len(rows)*100:4.1f}%  "
              f"boxed {sum(r['has_box'] for r in rows)/len(rows)*100:5.1f}%  ({(time.time()-t1)/60:.1f} min)")

shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", "results")
print("\ndownload: results.zip, grpo_a_adapter.zip, grpo_a_logs.zip")
'''),
    ]
    save("06-grpo-base.ipynb", cells)


if __name__ == "__main__":
    build_05()
    build_06()
