import os
import pickle

from rf_extract import answers_match, extract_answer, gsm8k_gold, has_boxed, user_prompt


def load_eval_set(name, n=None):
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"question": r["question"], "gold": gsm8k_gold(r["answer"])} for r in ds]
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": r["problem"], "gold": r["answer"]} for r in ds]
    else:
        raise ValueError(name)
    return items[:n] if n else items


def chat_prompts(tokenizer, items, style):
    return [tokenizer.apply_chat_template([{"role": "user", "content": user_prompt(it["question"], style)}],
                                          tokenize=False, add_generation_prompt=True)
            for it in items]


def rows_from_outputs(items, outputs):
    rows = []
    for it, out in zip(items, outputs):
        c = out.outputs[0]
        rows.append({"question": it["question"], "gold": it["gold"], "resp": c.text,
                     "n_new_tokens": len(c.token_ids),
                     "truncated": int(c.finish_reason == "length"),
                     "has_box": has_boxed(c.text)})
    return rows


def quick_acc(rows):
    return sum(answers_match(extract_answer(r["resp"]), r["gold"]) for r in rows) / max(len(rows), 1)


def load_result(path, config):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"config": config}


def save_result(path, result):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp, path)
