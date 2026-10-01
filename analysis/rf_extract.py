import re

BOXED_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def user_prompt(question, style="boxed"):
    if style == "boxed":
        return question + BOXED_SUFFIX
    if style == "plain":
        return question
    raise ValueError(style)


def _boxed_spans(text):
    key = r"\boxed{"
    spans, i = [], 0
    while True:
        j = text.find(key, i)
        if j == -1:
            break
        k = j + len(key)
        start, depth = k, 1
        while k < len(text) and depth > 0:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        spans.append(text[start:k - 1])
        i = k
    return spans


def _unwrap_text(s):
    return re.sub(r"\\(?:text|textbf|mbox|mathrm)\{([^{}]*)\}", r"\1", s)


def norm(x):
    if x is None:
        return None
    s = str(x).strip()
    s = _unwrap_text(s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace(" ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.rstrip(".").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def extract_boxed(text):
    boxes = _boxed_spans(text)
    return norm(boxes[-1]) if boxes else None


def extract_answer(text):
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return norm(nums[-1]) if nums else None


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _math_verify(pred, gold):
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    gf = _to_float(g)
    if gf is not None:
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[a-zA-Z]*", p)
        pf = _to_float(m.group(1)) if m else None
        return pf is not None and abs(pf - gf) < 1e-6
    return _math_verify(p, g)


def gsm8k_gold(answer_field):
    return norm(answer_field.split("####")[-1])


def has_boxed(text):
    return bool(_boxed_spans(text))
