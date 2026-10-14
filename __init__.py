"""effortless-hermes — per-prompt reasoning effort, picked by a fast judge.

A Hermes port of HeyCubit/effortless (MIT). An `llm_request` middleware reads the
next user message once per turn, asks a judge (Jev/TypeSafe, a local
OpenAI-compatible model, or any custom API) for the effort, and rewrites the
effort carrier in the provider request. Only the effort for that one request
changes — the conversation, model, and prompt-cache prefix stay untouched.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from . import judge as J

logger = logging.getLogger("hermes.plugins.effortless-hermes")

_MAX_CONTEXT_MSGS = 6
_MAX_CONTEXT_CHARS = 1600
_RECENT_VERDICTS = 50

_lock = threading.Lock()
_last_turn: dict[str, str] = {}          # session_id -> turn_id already judged
_last_effort: dict[str, str] = {}        # session_id -> effort applied this turn (reapplied on retries)
_plugin_ctx: Any = None                  # for the /effortless command


# ── message helpers ──────────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _messages(request: dict) -> list:
    msgs = request.get("messages")
    if not isinstance(msgs, list):
        msgs = request.get("input")
    return msgs if isinstance(msgs, list) else []


def _last_user_message(msgs: list) -> str:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text_of(m.get("content"))
    return ""


def _recent_context(msgs: list) -> str:
    lines = []
    for m in msgs[-_MAX_CONTEXT_MSGS:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        text = _text_of(m.get("content")).strip()
        if not text and isinstance(m.get("tool_calls"), list):
            text = f"[{len(m['tool_calls'])} tool call(s)]"
        if text:
            lines.append(f"{role}: {text[:300]}")
    context = "\n".join(lines)
    return context[-_MAX_CONTEXT_CHARS:]


# ── effort carriers ──────────────────────────────────────────────────────────

def _read_current_effort(request: dict) -> Optional[str]:
    v = request.get("reasoning_effort")
    if isinstance(v, str) and v:
        return v.lower()
    eb = request.get("extra_body")
    if isinstance(eb, dict):
        r = eb.get("reasoning")
        if isinstance(r, dict) and isinstance(r.get("effort"), str):
            return r["effort"].lower()
    r = request.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str):
        return r["effort"].lower()
    return None


def _write_effort(request: dict, effort: str) -> bool:
    """Rewrite every effort carrier already present. Never invents a new one."""
    wrote = False
    if isinstance(request.get("reasoning_effort"), str):
        request["reasoning_effort"] = effort
        wrote = True
    eb = request.get("extra_body")
    if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
        eb["reasoning"]["effort"] = effort
        eb["reasoning"]["enabled"] = True
        wrote = True
    r = request.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str):
        r["effort"] = effort
        wrote = True
    return wrote


# ── verdict log (durable, profile-scoped) ────────────────────────────────────

def _record(ctx_obj: Any, verdict: Optional[J.Verdict], prompt: str, note: str) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "effort": verdict.effort if verdict else None,
        "by": verdict.by if verdict else None,
        "sure": verdict.sure if verdict else None,
        "why": verdict.why if verdict else note,
        "ms": verdict.ms if verdict else 0,
        "prompt": prompt[:120],
        "note": note,
    }
    try:
        state = ctx_obj.state
        history = state.get("verdicts") or []
        history.append(entry)
        state.set("verdicts", history[-_RECENT_VERDICTS:])
    except Exception:
        logger.debug("effortless-hermes: could not persist verdict", exc_info=True)


def _cfg(ctx_obj: Any, key: str, default: Any) -> Any:
    try:
        value = ctx_obj.get_config(key, default)
    except Exception:
        value = default
    return default if value is None or value == "" else value


# ── the judge pipeline ───────────────────────────────────────────────────────

def _run_judges(cfg: dict, prompt: str, current_effort: Optional[str],
                context: str) -> tuple[Optional[J.Verdict], str]:
    """Primary judge, then the fallback when the primary is unsure/unavailable."""
    primary = str(cfg.get("judge_backend") or "jev").lower()
    fallback = str(cfg.get("fallback_backend") or "local").lower()
    errors: list[str] = []

    def backend(name: str) -> Optional[J.Verdict]:
        if name in ("none", ""):
            return None
        if name == "jev":
            key = J.find_typesafe_key()
            if not key:
                errors.append("jev: no TYPESAFE_API_KEY")
                return None
            try:
                return J.ask_jev(cfg, prompt, current_effort, context, key)
            except Exception as exc:
                errors.append(str(exc))
                return None
        if name in ("local", "custom"):
            try:
                return J.ask_openai_compat(cfg, name, prompt, current_effort, context)
            except Exception as exc:
                errors.append(str(exc))
                return None
        errors.append(f"unknown backend {name}")
        return None

    verdict = backend(primary)
    if verdict is None or verdict.unsure:
        second = backend(fallback)
        if second is not None and not second.unsure:
            return second, ("; ".join(errors) if verdict is None else
                            f"{primary} unsure -> {fallback}")
    if verdict is not None:
        return verdict, "; ".join(errors)
    return None, "; ".join(errors) or "no judge answered"


def on_llm_request(request: dict, **ctx: Any) -> Optional[dict]:
    """`llm_request` middleware: judge once per user turn, rewrite the effort carrier."""
    try:
        return _on_llm_request(request, **ctx)
    except Exception:
        logger.debug("effortless-hermes: middleware error (request unchanged)", exc_info=True)
        return None


def _on_llm_request(request: dict, **ctx: Any) -> Optional[dict]:
    global _plugin_ctx
    ctx_obj = _plugin_ctx
    if ctx_obj is None:
        return None
    if not _cfg(ctx_obj, "enabled", True):
        return None

    session_id = str(ctx.get("session_id") or "?")
    turn_id = str(ctx.get("turn_id") or "")
    with _lock:
        if _last_turn.get(session_id) == turn_id and turn_id:
            # already judged this turn; a retry rebuilt the request, so reapply
            cached = _last_effort.get(session_id)
            if cached and _write_effort(request, cached):
                return {"request": request, "source": "effortless-hermes",
                        "reason": f"reapply {cached} (retried request)"}
            return None  # tool-loop follow-ups keep the request as built
        _last_turn[session_id] = turn_id

    msgs = _messages(request)
    if not msgs:
        return None
    prompt = _last_user_message(msgs)
    if not prompt.strip():
        return None
    context = _recent_context(msgs)
    current_effort = _read_current_effort(request)
    model = str(request.get("model") or ctx.get("model") or "")
    cfg = {key: _cfg(ctx_obj, key, default) for key, default in [
        ("enabled", True), ("judge_backend", "jev"), ("fallback_backend", "local"),
        ("jev_url", "https://api.typesafe.ai/v1/systemone"), ("jev_model", "jev-latest"),
        ("typesafe_key", ""), ("local_base_url", "http://127.0.0.1:8080/v1"),
        ("local_model", ""), ("custom_base_url", ""), ("custom_model", ""),
        ("custom_api_key", ""), ("judge_timeout_seconds", 3.0),
        ("effort_floor", "low"), ("effort_ceiling", "max"), ("effort_bias", 0),
    ]}

    floor = str(cfg["effort_floor"]); ceiling = str(cfg["effort_ceiling"])
    bias = int(cfg["effort_bias"] or 0)

    # 1. named level in the message wins, no judge
    named = J.named_effort(prompt)
    if named:
        effort = J.apply_bounds(named, floor, ceiling, bias)
        effort = J.clamp_to_wire(effort, model)
        if _write_effort(request, effort):
            _last_effort[session_id] = effort
            _record(ctx_obj, None, prompt, f"named in message -> {effort}")
            return {"request": request, "source": "effortless-hermes",
                    "reason": f"named: {effort}"}
        return None

    # 2. short follow-up keeps the current effort
    if current_effort and J.keeps_effort(prompt, context):
        _last_effort[session_id] = current_effort
        _record(ctx_obj, None, prompt, "follow-up, kept effort")
        return None

    # 3. ask the judges
    verdict, note = _run_judges(cfg, prompt, current_effort, context)
    if verdict is None:
        _record(ctx_obj, None, prompt, note)
        return None

    effort = J.apply_bounds(verdict.effort, floor, ceiling, bias)
    effort = J.clamp_to_wire(effort, model)
    if effort == current_effort:
        _last_effort[session_id] = effort
        _record(ctx_obj, verdict, prompt, "same as current, unchanged")
        return None
    if _write_effort(request, effort):
        _last_effort[session_id] = effort
        _record(ctx_obj, verdict, prompt, f"-> {effort} (was {current_effort})")
        reason = f"{verdict.by}: {effort}"
        if verdict.why and verdict.why != "unsure, keeping":
            reason += f" ({verdict.why})"
        return {"request": request, "source": "effortless-hermes", "reason": reason}
    _record(ctx_obj, verdict, prompt, "no effort carrier on this provider; skipped")
    return None


# ── /effortless command ──────────────────────────────────────────────────────

def _command_effortless(raw_args: str = "") -> str:
    arg = (raw_args or "").strip().lower()
    ctx_obj = _plugin_ctx
    if ctx_obj is None:
        return "effortless-hermes: not loaded."
    if arg in ("test", "check"):
        cfg = {k: _cfg(ctx_obj, k, d) for k, d in [("judge_backend", "jev"),
                ("fallback_backend", "local")]}
        out = [f"primary judge: {cfg['judge_backend']}, fallback: {cfg['fallback_backend']}"]
        sample = "rename one variable in a single file"
        try:
            v, note = _run_judges(cfg, sample, "medium", "")
            out.append(f"test prompt: {sample!r}")
            out.append(f"verdict: {v!r}" if v else f"no verdict: {note}")
        except Exception as exc:
            out.append(f"test failed: {exc}")
        return "\n".join(out)
    history = []
    try:
        history = (ctx_obj.state.get("verdicts") or [])[-10:]
    except Exception:
        pass
    if not history:
        return ("effortless-hermes: no verdicts recorded yet. Next prompt will be judged. "
                "Try '/effortless test' to check the judges.")
    lines = ["effortless-hermes — last verdicts (newest last):"]
    for h in history:
        lines.append(f"  {h['ts']}  {str(h['effort'] or '-'):6} {str(h['by'] or '-'):6} "
                     f"{h['why']}  | {h['prompt'][:60]}")
    return "\n".join(lines)


# ── registration ─────────────────────────────────────────────────────────────

def register(ctx) -> None:
    global _plugin_ctx
    _plugin_ctx = ctx
    ctx.register_middleware("llm_request", on_llm_request)
    try:
        ctx.register_command("effortless", _command_effortless,
                             description="effortless-hermes: show recent effort verdicts; "
                                         "'/effortless test' checks the judges",
                             argument_mode="text")
    except Exception:
        logger.debug("effortless-hermes: command registration failed", exc_info=True)
    logger.info("effortless-hermes registered (judge=%s fallback=%s)",
                ctx.get_config("judge_backend", "jev"), ctx.get_config("fallback_backend", "local"))
