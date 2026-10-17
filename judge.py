"""effortless-hermes judge backends: Jev (TypeSafe), local OpenAI-compat, custom API.

Stdlib only (urllib) — plugin must survive PM-managed env rebuilds without pip deps.
Prompts ported from HeyCubit/effortless (MIT), adapted for Hermes' effort ladder.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Optional

EFFORTS = ["low", "medium", "high", "xhigh", "max"]

# ── Hermes wire vocabularies (from agent.reasoning_effort; applied post-judge) ──
_WIRE_VOCAB: list[tuple[str, tuple[str, ...]]] = [
    # (model-substring-lower, vocabulary) — first match wins
    ("glm-5.3", ("low", "medium", "high", "max")),
    ("glm-5.2", ("high", "max")),
    ("deepseek", ("low", "medium", "high", "max")),
    ("k3", ("low", "high", "max")),
    ("glm-4.5", ("low", "medium", "high")),
]
_DEFAULT_VOCAB = ("low", "medium", "high", "xhigh", "max")
_WIRE_OVERRIDES = {  # nearest-accepted map where the ladder has gaps
    "glm-5.3": {"xhigh": "max"},
    "glm-5.2": {"low": "high", "medium": "high", "xhigh": "max"},
    "k3": {"medium": "high", "xhigh": "max"},
}
# Providers that accept exactly low/high/max (the restricted triple). medium maps
# UP to high, not down to low: the judge called the prompt "normal work"
# (a few tool calls) — low means "no tools, no planning", which under-powers it.
# Same convention as the k3 mapping. xhigh maps to max.
_TRIPLE_VOCAB = ("low", "high", "max")
_TRIPLE_OVERRIDES = {"medium": "high", "xhigh": "max"}

JUDGE_SYSTEM = """You choose which reasoning effort an agentic assistant (it reads files, runs tools and edits things, not only code) should use for the user's next message. Pick the cheapest effort that will still do the job well.

Effort: low for quick answers, medium for normal work, high for hard problems, xhigh or max only for very hard ones.

Effort is relative to the model in use ("Current" names it): a stronger model needs less effort for the same job. So for one and the same task pick one step lower on a strong model than on a weaker one, and one step higher on a weak model for hard work.

Judge the SCOPE and the amount of work, not whether it is code. A short message can ask for a lot: "go through my whole drive and clean it up", "review the entire repo", "migrate everything" are big, multi-step, tool-heavy jobs where mistakes are costly: never low, usually high. Low is only for answers that need no tools and no planning.

If the message answers a question in the assistant's last reply (picks an option, says which one, confirms a plan), judge only the work that answer starts, as the reply describes it, not the length of the answer and not the work done before the question: "B" can mean "build the complicated section B" (high), while "yes" or "no" to one small action ("should I archive this?", "delete the old ones too?") is low, and picking a value for one setting (a log level, a colour, a font) is low. But picking which way to build something ("option 1", "the same shapes", "B") starts that build: judge the build. Approving a whole plan or several steps takes the effort of that plan. A plain "yes", "ok" or "do it" after "Is that OK? Then I'll build X" or "Should I build X?" approves building X: judge X, not the word.

A message that pushes back on, corrects or adds to a plan or claim under discussion continues that work: keep at least the effort that work had, never drop to low for it. Thanks, praise or a closing remark with no new request is low. A question about how to do something, or about effort itself, that needs no tools is low.

If the person asks for deep thought ("think hard", "ultrathink", "be thorough"), pick at least high; if they ask for a quick answer, pick low.

If the message is a short follow-up to ongoing work ("yes", "go", "ok", "continue", or the same in any language), keep the current effort.

Reply with JSON only: {"effort":"low|medium|high|xhigh|max","sure":0.0-1.0 how sure you are of the effort,"why":"at most 6 words, in the user's language"}"""

JEV_TASK = (
    "Choose the reasoning effort an agentic assistant should use for the next user message. It reads files, "
    "runs tools and edits things, not only code. Pick the cheapest that still does the job well. Judge the scope "
    "and the amount of work, not whether it is code: \"go through my whole drive and clean it up\" is a big "
    "tool-heavy job. Effort is relative to current_model: a stronger model needs less for the same job. "
    "A short follow-up (\"yes\", \"go\", \"ok\", in any language) keeps the current effort. When the message answers a "
    "question in the assistant's last reply (picks an option), judge only the work that answer starts, not its length "
    "and not the work before the question: yes/no to one small action, or picking a value for one setting (a log level, "
    "a colour), is low; picking which way to build something starts that build, so judge the build; approving a whole "
    "plan takes the effort of that plan; a plain yes to \"Then I will build X\" means judge X. Pushing back on or adding "
    "to a plan under discussion continues that work: keep its effort, never low. Thanks or a closing remark with no new "
    "request is low. A question about how to do something that needs no tools is low. \"think hard\", \"ultrathink\" or "
    "\"be thorough\" means at least high; \"quick question\" means low."
)

JEV_EFFORT_CRITERIA = {
    "low": "quick answer, no tools, no planning",
    "medium": "normal work: a few tool calls or edits",
    "high": "hard or large multi-step job: many tool calls, costly mistakes",
    "xhigh": "very hard: a long investigation",
    "max": "the hardest research-level work",
}


class Verdict:
    def __init__(self, effort: str, why: str, by: str, sure: Optional[float] = None,
                 unsure: bool = False, ms: int = 0):
        self.effort, self.why, self.by, self.sure, self.unsure, self.ms = effort, why, by, sure, unsure, ms

    def __repr__(self):
        return f"<Verdict {self.effort} by={self.by} sure={self.sure} unsure={self.unsure} why={self.why!r}>"


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8", "replace"))


def parse_verdict_text(text: str) -> Optional[tuple[str, Optional[float], str]]:
    """`{effort, sure, why}` out of a judge reply (JSON, possibly embedded in prose)."""
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return None
    effort = str(raw.get("effort", "")).lower().strip()
    if effort not in EFFORTS:
        return None
    sure = raw.get("sure")
    sure = float(sure) if isinstance(sure, (int, float)) and 0.0 <= sure <= 1.0 else None
    why = str(raw.get("why") or "")[:80]
    return effort, sure, why


# ── Jev (TypeSafe systemone) ─────────────────────────────────────────────────

def find_typesafe_key() -> Optional[str]:
    """Plugin-scoped secret first, then the shared TYPESAFE_API_KEY — both through
    Hermes' secret scope (profile-aware, never a direct .env or os.environ read)."""
    try:
        from agent.secret_scope import get_secret_str
        for name in ("EFFORTLESS_HERMES_TYPESAFE_KEY", "TYPESAFE_API_KEY"):
            val = get_secret_str(name, "").strip()
            if val:
                return val
    except Exception:
        pass
    return None


def ask_jev(cfg: dict, prompt: str, current_effort: Optional[str], context: str,
            key: str) -> Optional[Verdict]:
    import time
    t0 = time.monotonic()
    payload = {
        "model": cfg.get("jev_model") or "jev-latest",
        "state": {
            "task": JEV_TASK,
            "current_model": None,
            "current_effort": current_effort,
            "recent_conversation": context[:4000] if context else "",
            "next_message": prompt[:4000],
        },
        "questions": {
            "effort": {"type": "choice", "instructions": "Which effort fits the next message?",
                       "criteria": JEV_EFFORT_CRITERIA},
        },
    }
    try:
        data = _post_json(cfg.get("jev_url") or "https://api.typesafe.ai/v1/systemone",
                          payload, {"Authorization": f"Bearer {key}"},
                          float(cfg.get("judge_timeout_seconds") or 3.0))
    except Exception as exc:
        raise RuntimeError(f"jev unreachable: {exc}") from exc
    answers = (data.get("answers") or (data.get("result") or {}).get("answers") or {})
    effort = (answers.get("effort") or {}).get("choice")
    if not isinstance(effort, str) or effort.lower() not in EFFORTS:
        return None
    effort = effort.lower()
    sure = (answers.get("effort") or {}).get("confidence")
    if sure is None:
        sure = ((answers.get("effort") or {}).get("probabilities") or {}).get(effort)
    sure = float(sure) if isinstance(sure, (int, float)) else None
    unsure = current_effort is not None and sure is not None and sure < 0.5
    return Verdict(current_effort if unsure else effort,
                   "unsure, keeping" if unsure else (f"{round(sure * 100)}% sure" if sure is not None else ""),
                   "jev", sure=sure, unsure=unsure, ms=int((time.monotonic() - t0) * 1000))


# ── OpenAI-compatible judge (local model or custom API) ──────────────────────

def _local_model_name(cfg: dict) -> str:
    return (cfg.get("local_model") or "").strip() or "local-model"


def ask_openai_compat(cfg: dict, which: str, prompt: str, current_effort: Optional[str],
                      context: str) -> Optional[Verdict]:
    import time
    t0 = time.monotonic()
    base = (cfg.get(f"{which}_base_url") or "").strip().rstrip("/")
    model = (cfg.get(f"{which}_model") or "").strip() or (
        _local_model_name(cfg) if which == "local" else "")
    if not base or not model:
        raise RuntimeError(f"{which} judge not configured (base_url/model)")
    key = ""
    if which == "custom":
        try:
            from agent.secret_scope import get_secret_str
            key = get_secret_str("EFFORTLESS_HERMES_CUSTOM_API_KEY", "").strip()
        except Exception:
            key = ""
    question = "\n\n".join(filter(None, [
        f"Current: {current_effort}" if current_effort else "Current: none",
        f"Recent conversation:\n{context[:1500]}" if context else "",
        f"Next message:\n{prompt[:2000]}",
    ]))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": question}],
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        data = _post_json(f"{base}/chat/completions", payload, headers,
                          max(float(cfg.get("judge_timeout_seconds") or 3.0), 8.0))
    except Exception as exc:
        raise RuntimeError(f"{which} judge unreachable: {exc}") from exc
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except Exception:
        return None
    parsed = parse_verdict_text(text)
    if not parsed:
        return None
    effort, sure, why = parsed
    unsure = current_effort is not None and sure is not None and sure < 0.5
    return Verdict(current_effort if unsure else effort,
                   "unsure, keeping" if unsure else why,
                   which, sure=sure, unsure=unsure, ms=int((time.monotonic() - t0) * 1000))


# ── Wire clamping (vocabulary per model, then floor/ceiling/bias) ────────────

# Surplus Intelligence (api.surplusintelligence.ai) accepts glm-5.3's full
# scale on plain one-shot chat, but rejects anything but low/high/max once the
# request is agentic (tools attached or a long agent-style history) — and the
# same 400 fires for other models it relays (live-verified 2026-10-17 on
# gpt-6.1-sol: "'reasoning_effort' must be one of: 'low', 'high', 'max'",
# request id 01M542VPWVB11AQ0SNWYMGS1RT). Hermes requests are always agentic
# (tools on every turn), so any Surp-hosted agentic request — any model —
# clamps to the restricted triple.
_SURP_URL_NEEDLE = "surplusintelligence"


def _is_agentic_request(tools: Any) -> bool:
    return bool(tools)


def wire_vocabulary(model: str, base_url: Optional[str] = None,
                    tools: Any = None) -> tuple[str, ...]:
    m = (model or "").lower()
    url = (base_url or "").lower()
    if _SURP_URL_NEEDLE in url:
        if _is_agentic_request(tools) or "glm-5.3" in m:
            return _TRIPLE_VOCAB
    elif not url and "glm-5.3" in m and _is_agentic_request(tools):
        # no provider info and an agentic glm-5.3 request: be safe and use the
        # restricted triple (real middleware calls always carry base_url, so
        # direct-z.ai requests with their full scale are not affected).
        return _TRIPLE_VOCAB
    for needle, vocab in _WIRE_VOCAB:
        if needle in m:
            return vocab
    return _DEFAULT_VOCAB


def clamp_to_wire(effort: str, model: str, base_url: Optional[str] = None,
                  tools: Any = None) -> str:
    vocab = wire_vocabulary(model, base_url, tools)
    if effort in vocab:
        return effort
    if vocab == _TRIPLE_VOCAB and effort in _TRIPLE_OVERRIDES:
        # restricted providers: keep the verdict's intent (medium -> high),
        # never silently downgrade judged "normal work" to a no-tools low.
        return _TRIPLE_OVERRIDES[effort]
    m = (model or "").lower()
    for needle, overrides in _WIRE_OVERRIDES.items():
        if needle in m and effort in overrides:
            mapped = overrides[effort]
            if mapped in vocab:
                return mapped
    # nearest weaker level
    ladder = [e for e in EFFORTS if e in vocab]
    idx = EFFORTS.index(effort)
    for e in reversed(EFFORTS[:idx]):
        if e in vocab:
            return e
    return ladder[0]


def apply_bounds(effort: str, floor: str, ceiling: str, bias: int) -> str:
    ladder = EFFORTS
    i = ladder.index(effort) if effort in ladder else ladder.index("medium")
    i = max(i + bias, 0) if bias else i
    i = max(i, ladder.index(floor) if floor in ladder else 0)
    i = min(i, ladder.index(ceiling) if ceiling in ladder else len(ladder) - 1)
    return ladder[i]


# ── Prompt heuristics ported from effortless ─────────────────────────────────

_LEVEL_WORD = r"(low|medium|high|xhigh|x-high|extra[- ]high|max|maximum)"
_NAMED_LEVEL = re.compile(
    rf"\b{_LEVEL_WORD}\s+effort\b|\beffort(?:\s+level)?\s*(?:[:=]|to|at|of|on)?\s*{_LEVEL_WORD}\b|\bxhigh\b",
    re.I)
_LEVEL_OF = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh",
             "x-high": "xhigh", "extra high": "xhigh", "extra-high": "xhigh",
             "max": "max", "maximum": "max"}


def named_effort(text: str) -> Optional[str]:
    """The level a prompt asks for by name ("max effort", "effort: high"), or None."""
    for m in _NAMED_LEVEL.finditer(text or ""):
        before = text[max(0, m.start() - 20):m.start()].lower()
        if re.search(r"\b(not|no|don'?t|never|without|instead of)\b", before):
            continue
        return _LEVEL_OF[(m.group(1) or m.group(2) or "xhigh").lower().replace("  ", " ")]
    return None


def is_follow_up(text: str) -> bool:
    t = (text or "").strip()
    return 0 < len(t) <= 12 and len(t.split()) <= 2


def ends_on_question(context: str) -> bool:
    reply = (context or "").split("\nassistant: ")[-1]
    return "assistant: " in (context or "") and "?" in reply[-400:]


def keeps_effort(message: str, context: str) -> bool:
    return is_follow_up(message) and not ends_on_question(context)
