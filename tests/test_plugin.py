"""Offline tests for effortless-hermes. Run: python3 -m pytest tests/ -q
(or python3 tests/test_plugin.py). No network, no Hermes install required for
the logic tests; live-judge tests are skipped unless their env is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("effortless_judge", PLUGIN / "judge.py")
core = _load("effortless_core", PLUGIN / "__init__.py")


# ── heuristics ───────────────────────────────────────────────────────────────

def test_named_effort():
    assert J.named_effort("please use max effort on this") == "max"
    assert J.named_effort("effort: high") == "high"
    assert J.named_effort("set effort to xhigh now") == "xhigh"
    assert J.named_effort("don't use max effort") is None
    assert J.named_effort("a medium sized image") is None
    assert J.named_effort("") is None


def test_follow_up():
    assert J.is_follow_up("go")
    assert J.is_follow_up("ok")
    assert J.is_follow_up("yes")
    assert J.is_follow_up("not really though") is False  # >12 chars
    assert J.is_follow_up("please continue with the work") is False


def test_keeps_effort():
    assert J.keeps_effort("go", "user: do X\nassistant: Working on it.")
    # "ok" answering a question is NOT a plain follow-up
    assert not J.keeps_effort("ok", "user: hi\nassistant: Should I archive it?")


def test_ends_on_question():
    assert J.ends_on_question("user: x\nassistant: shall I continue?")
    assert not J.ends_on_question("user: x\nassistant: done.")


# ── verdict parsing ──────────────────────────────────────────────────────────

def test_parse_verdict_text():
    assert J.parse_verdict_text('{"effort":"high","sure":0.9,"why":"refactor"}') == ("high", 0.9, "refactor")
    assert J.parse_verdict_text('Sure! {"effort":"low","sure":0.8,"why":"quick"}') == ("low", 0.8, "quick")
    assert J.parse_verdict_text('{"effort":"bogus"}') is None
    assert J.parse_verdict_text("no json here") is None
    assert J.parse_verdict_text('{"effort":"low","sure":"weird"}') == ("low", None, "")


# ── wire clamping ────────────────────────────────────────────────────────────

def test_clamp_to_wire():
    assert J.clamp_to_wire("xhigh", "glm-5.3") == "max"
    assert J.clamp_to_wire("low", "glm-5.2") == "high"
    assert J.clamp_to_wire("medium", "kimi-k3-thinking") == "high"
    assert J.clamp_to_wire("xhigh", "k3") == "max"
    assert J.clamp_to_wire("low", "some-model") == "low"
    assert J.clamp_to_wire("xhigh", "some-model") == "xhigh"
    assert J.clamp_to_wire("low", "deepseek-v4.1") == "low"


def test_wire_vocabulary():
    assert J.wire_vocabulary("GLM-5.3") == ("low", "medium", "high", "max")
    assert "xhigh" in J.wire_vocabulary("unknown-model")


def test_apply_bounds():
    assert J.apply_bounds("low", "medium", "max", 0) == "medium"    # floor
    assert J.apply_bounds("max", "low", "high", 0) == "high"        # ceiling
    assert J.apply_bounds("medium", "low", "max", -2) == "low"      # bias cheaper
    assert J.apply_bounds("low", "low", "max", 2) == "high"         # bias smarter


# ── effort carriers ──────────────────────────────────────────────────────────

def test_read_write_effort_carriers():
    req = {"reasoning_effort": "medium", "messages": []}
    assert core._read_current_effort(req) == "medium"
    assert core._write_effort(req, "high")
    assert req["reasoning_effort"] == "high"

    req2 = {"extra_body": {"reasoning": {"enabled": True, "effort": "low"}}}
    assert core._read_current_effort(req2) == "low"
    core._write_effort(req2, "max")
    assert req2["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}

    req3 = {"reasoning": {"effort": "medium"}}
    assert core._read_current_effort(req3) == "medium"

    # no carrier -> never invents one
    req4 = {"model": "x", "messages": []}
    assert core._read_current_effort(req4) is None
    assert core._write_effort(req4, "high") is False


def test_message_extraction():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "image question?"},
                                     {"type": "image_url", "image_url": {"url": "x"}}]},
    ]
    assert core._last_user_message(msgs) == "image question?"
    ctx = core._recent_context(msgs)
    assert "hello there" in ctx and "image question?" in ctx


# ── middleware decision paths (offline; judges stubbed) ─────────────────────

class FakeState:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class FakeCtx:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self._state = FakeState()

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    @property
    def state(self):
        return self._state


def _req(prompt="do a big refactor of everything", effort="medium"):
    return {"model": "glm-5.3", "reasoning_effort": effort,
            "messages": [{"role": "user", "content": prompt}]}


def _run_middleware(fake_ctx, request, turn="t1", session="s1", base_url=""):
    core._plugin_ctx = fake_ctx
    core._last_turn.pop(session, None)
    core._last_effort.pop(session, None)
    core._last_model.pop(session, None)
    return core.on_llm_request(request, session_id=session, turn_id=turn,
                               model="glm-5.3", provider="x", base_url=base_url,
                               api_mode="chat_completions", api_call_count=1)


def test_middleware_named_level_skips_judge(monkeypatch_none=None):
    ctx = FakeCtx({"judge_backend": "jev"})
    req = _req("please use max effort for this migration")
    res = _run_middleware(ctx, req)
    assert req["reasoning_effort"] == "max"
    assert res and res["reason"] == "named: max"


def test_middleware_followup_keeps():
    ctx = FakeCtx({"judge_backend": "jev"})
    req = _req("go")
    res = _run_middleware(ctx, req)
    assert res is None and req["reasoning_effort"] == "medium"
    assert ctx.state.get("verdicts")[-1]["why"] == "follow-up, kept effort"


def test_middleware_disabled():
    ctx = FakeCtx({"enabled": False})
    req = _req()
    assert _run_middleware(ctx, req) is None


def test_middleware_all_judges_fail_keeps():
    ctx = FakeCtx({"judge_backend": "jev", "fallback_backend": "local",
                   "local_base_url": "http://127.0.0.1:1/v1"})  # nothing listens
    req = _req()
    res = _run_middleware(ctx, req)
    assert res is None and req["reasoning_effort"] == "medium"
    note = ctx.state.get("verdicts")[-1]["why"]
    assert "jev" in note or "no judge" in note or "unreachable" in note


def test_middleware_judges_once_per_turn():
    ctx = FakeCtx({"judge_backend": "local", "local_base_url": "http://127.0.0.1:1/v1",
                   "fallback_backend": "none"})
    req = _req()
    _run_middleware(ctx, req, turn="t1", session="sx")
    calls = []
    orig = core._run_judges
    core._run_judges = lambda *a, **k: calls.append(1) or (None, "stub")
    res = core.on_llm_request(req, session_id="sx", turn_id="t1", model="glm-5.3")
    assert not calls  # same turn: no second judge call
    core._run_judges = orig


def test_middleware_verdict_applied(monkeypatch_none=None):
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        res = _run_middleware(ctx, req)
        assert req["reasoning_effort"] == "high"
        assert res and "high" in res["reason"]
    finally:
        core._run_judges = orig


def test_middleware_unsure_verdict_falls_to_fallback_result():
    # primary unsure -> _run_judges already resolves; here unsure keeps current
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("medium", "unsure, keeping", "jev", sure=0.4, unsure=True)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "fallback also unsure")
    try:
        req = _req("a tricky little thing", effort="medium")
        res = _run_middleware(ctx, req)
        assert res is None and req["reasoning_effort"] == "medium"
    finally:
        core._run_judges = orig


def test_no_carrier_no_crash():
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("high", "big", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = {"model": "some-model", "messages": [{"role": "user", "content": "hi"}]}
        res = _run_middleware(ctx, req)
        assert res is None  # nothing to rewrite -> skipped, never invented
    finally:
        core._run_judges = orig




# ── v0.3: model_map, post_api_request, stats ─────────────────────────────────

def test_model_map_fresh_chat_applies():
    ctx = FakeCtx({"model_map": {"high": "strong-model", "low": "cheap-model"}})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()  # one user message = fresh chat
        res = _run_middleware(ctx, req)
        assert req["model"] == "strong-model"
        assert req["reasoning_effort"] == "high"
        assert "model: strong-model" in res["reason"]
    finally:
        core._run_judges = orig


def test_model_map_established_chat_skips():
    ctx = FakeCtx({"model_map": {"high": "strong-model"}})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        req["messages"] = [{"role": "user", "content": "first"},
                           {"role": "assistant", "content": "hi"},
                           {"role": "user", "content": "do a big refactor of everything"}]
        res = _run_middleware(ctx, req)
        assert req["model"] == "glm-5.3"  # unchanged: cache safety
        assert "model" not in (res["reason"] if res else "")
    finally:
        core._run_judges = orig


def test_model_map_always_overrides():
    ctx = FakeCtx({"model_map": {"high": "strong-model"}, "model_map_always": True})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        req["messages"] = [{"role": "user", "content": "first"},
                           {"role": "assistant", "content": "hi"},
                           {"role": "user", "content": "do a big refactor of everything"}]
        _run_middleware(ctx, req)
        assert req["model"] == "strong-model"
    finally:
        core._run_judges = orig


def test_model_map_reapplied_on_retry():
    ctx = FakeCtx({"model_map": {"high": "strong-model"}})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        _run_middleware(ctx, req, turn="t9", session="s9")
        req2 = _req()  # rebuilt by a retry: model and effort must come back
        res = core.on_llm_request(req2, session_id="s9", turn_id="t9", model="glm-5.3")
        assert req2["model"] == "strong-model"
        assert req2["reasoning_effort"] == "high"
        assert res is not None
    finally:
        core._run_judges = orig


def test_model_map_noop_when_same_model():
    ctx = FakeCtx({"model_map": {"high": "glm-5.3"}})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        res = _run_middleware(ctx, req)
        assert req["model"] == "glm-5.3"
        assert not ctx.state.get("verdicts")[-1].get("model_override")
    finally:
        core._run_judges = orig


def test_post_api_request_records_first_call_only():
    ctx = FakeCtx()
    core._plugin_ctx = ctx
    core.on_post_api_request(session_id="sx", api_call_count=1,
                             model="glm-5.3", usage={"prompt_tokens": 12000, "total_tokens": 12500},
                             context_length=65536, api_duration=1.5)
    core.on_post_api_request(session_id="sx", api_call_count=2,
                             model="glm-5.3", usage={"prompt_tokens": 12200},
                             context_length=65536, api_duration=0.4)
    turns = ctx.state.get("turns")
    assert len(turns) == 1
    assert turns[0]["input_tokens"] == 12000
    assert turns[0]["context_length"] == 65536


def test_context_fill():
    assert core._context_fill({"input_tokens": 32768, "context_length": 65536}) == 50
    assert core._context_fill({"input_tokens": 65536, "context_length": 65536}) == 100
    assert core._context_fill({"input_tokens": 100}) is None
    assert core._context_fill(None) is None


def test_stats_output():
    ctx = FakeCtx()
    core._plugin_ctx = ctx
    ctx.state.set("verdicts", [
        {"ts": "t", "effort": "low", "by": "jev", "sure": 0.9, "why": "", "ms": 300, "prompt": "p", "note": ""},
        {"ts": "t", "effort": "high", "by": "jev", "sure": 0.8, "why": "", "ms": 400, "prompt": "p", "note": ""},
        {"ts": "t", "effort": None, "by": None, "sure": None, "why": "follow-up", "ms": 0, "prompt": "p", "note": ""},
    ])
    ctx.state.set("turns", [{"input_tokens": 20000, "context_length": 65536}])
    out = core._fmt_stats(ctx)
    assert "low×1" in out and "high×1" in out
    assert "jev×2" in out
    assert "avg 350 ms" in out
    assert "31% of window" in out
    assert "no cost-savings estimate" in out


def test_command_stats_route():
    ctx = FakeCtx()
    core._plugin_ctx = ctx
    out = core._command_effortless("stats")
    assert "stats" in out and "verdicts recorded" in out


# ── Surplus Intelligence glm-5.3: agentic routes accept low/high/max only ────
# Live-verified 2026-10-17: plain chat accepts xhigh/medium fine, but any
# request with `tools` (or a long agent-style history) 400s with
# "'reasoning_effort' must be one of: 'low', 'high', 'max'".

def test_clamp_to_wire_surp_glm53_agentic_accepts_only_lhm():
    assert J.clamp_to_wire("xhigh", "glm-5.3", tools=[{"type": "function"}]) == "max"
    # medium is illegal on Surp's agentic routes; it maps UP to high, not down
    # to low: the judge called the prompt "normal work" (a few tool calls) and
    # low means "no tools, no planning" — downgrading under-powers the prompt.
    assert J.clamp_to_wire("medium", "glm-5.3", tools=[{"type": "function"}]) == "high"
    assert J.clamp_to_wire("high", "glm-5.3", tools=[{"type": "function"}]) == "high"
    assert J.clamp_to_wire("low", "glm-5.3", tools=[{"type": "function"}]) == "low"
    assert J.clamp_to_wire("max", "glm-5.3", tools=[{"type": "function"}]) == "max"


def test_clamp_to_wire_surp_any_model_agentic():
    # the live 400 (2026-10-17) fired on gpt-6.1-sol, not glm: the restricted
    # triple applies to ANY model the provider relays once the request is agentic
    surp = "https://api.surplusintelligence.ai/v1"
    tools = [{"type": "function"}]
    assert J.wire_vocabulary("gpt-6.1-sol", base_url=surp, tools=tools) == ("low", "high", "max")
    assert J.clamp_to_wire("medium", "gpt-6.1-sol", base_url=surp, tools=tools) == "high"
    assert J.clamp_to_wire("xhigh", "gpt-6.1-sol", base_url=surp, tools=tools) == "max"
    assert J.clamp_to_wire("low", "gpt-6.1-sol", base_url=surp, tools=tools) == "low"
    # non-agentic, non-glm via surp: model table applies (unverified territory
    # stays open rather than guessed)
    assert J.wire_vocabulary("gpt-6.1-sol", base_url=surp) == J._DEFAULT_VOCAB
    # same model on a different provider is untouched by the surp rule
    assert J.wire_vocabulary("gpt-6.1-sol", base_url="https://api.openai.com/v1") == J._DEFAULT_VOCAB


def test_clamp_to_wire_surp_plain_chat_unchanged():
    # No tools: plain-chat vocab applies, medium stays verbatim.
    assert J.clamp_to_wire("medium", "glm-5.3") == "medium"


def test_wire_vocabulary_provider_rule():
    # base_url rules key off the provider, not the model, when present.
    assert J.wire_vocabulary("glm-5.3", base_url="https://api.surplusintelligence.ai/v1") == ("low", "high", "max")


def test_middleware_surp_agentic_clamps_medium():
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("medium", "normal", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()  # no tools: plain-chat vocab applies, medium stays
        _run_middleware(ctx, req)
        assert req["reasoning_effort"] == "medium"
        req2 = _req()
        req2["tools"] = [{"type": "function", "function": {"name": "f"}}]
        _run_middleware(ctx, req2, turn="t2", session="s2")
        assert req2["reasoning_effort"] == "high"  # agentic Surp: medium illegal -> high
    finally:
        core._run_judges = orig


def test_middleware_cached_effort_reapplied_clamps_on_retry():
    # Retry rebuilt the request; the cached effort must be re-clamped against
    # the rebuilt request (tools), not pasted verbatim (the 01M543N9... 400).
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("xhigh", "hard", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()  # plain chat: judged effort stored
        _run_middleware(ctx, req, turn="t1", session="sx")
        req2 = _req()
        req2["tools"] = [{"type": "function"}]  # agentic retry
        core.on_llm_request(req2, session_id="sx", turn_id="t1", model="glm-5.3")
        assert req2["reasoning_effort"] in ("low", "high", "max")  # never xhigh on agentic Surp
    finally:
        core._run_judges = orig


SURP = "https://api.surplusintelligence.ai/v1"
AG = [{"type": "function", "function": {"name": "f"}}]


def test_sanitize_config_medium_on_agentic_surp():
    # the live incident: config sets 'medium', provider rejects it. Even with
    # no verdict (judges down) and a follow-up prompt, the request must ship
    # with a legal effort.
    ctx = FakeCtx({"judge_backend": "jev", "fallback_backend": "local",
                   "local_base_url": "http://127.0.0.1:1/v1"})  # judges all down
    req = {"model": "gpt-6.1-sol", "reasoning_effort": "medium",
           "messages": [{"role": "user", "content": "ok"}], "tools": AG}
    res = _run_middleware(ctx, req, base_url=SURP)
    assert req["reasoning_effort"] == "high"  # sanitized, medium -> high
    assert res is not None and "sanitized" in res["reason"]


def test_sanitize_config_medium_no_carrier_untouched():
    # no effort carrier -> nothing invented, nothing broken
    ctx = FakeCtx({"judge_backend": "jev"})
    req = {"model": "gpt-6.1-sol",
           "messages": [{"role": "user", "content": "hi"}], "tools": AG}
    res = _run_middleware(ctx, req, base_url=SURP)
    assert "reasoning_effort" not in req


def test_sanitize_on_retry_of_unjudged_turn():
    # retry rebuilt the request; the turn was never judged (judge down) so no
    # cached effort exists — the sanitize alone must still fix the 400 carrier
    ctx = FakeCtx({"judge_backend": "jev", "fallback_backend": "local",
                   "local_base_url": "http://127.0.0.1:1/v1"})
    req = {"model": "gpt-6.1-sol", "reasoning_effort": "medium",
           "messages": [{"role": "user", "content": "what is 2+2"}], "tools": AG}
    res = _run_middleware(ctx, req, turn="t1", session="sr", base_url=SURP)
    assert req["reasoning_effort"] == "high"
    req2 = {"model": "gpt-6.1-sol", "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": "what is 2+2"}], "tools": AG}
    res2 = core.on_llm_request(req2, session_id="sr", turn_id="t1", model="gpt-6.1-sol",
                               base_url=SURP, api_mode="chat_completions")
    assert req2["reasoning_effort"] == "high"
    assert res2 is not None


def test_no_sanitize_when_effort_is_legal():
    # deepseek accepts medium; a surp base_url with a NON-agentic request and a
    # non-glm model must not touch a legal value
    ctx = FakeCtx({"judge_backend": "jev"})
    req = {"model": "deepseek-v4.1-flash", "reasoning_effort": "medium",
           "messages": [{"role": "user", "content": "hi"}]}
    _run_middleware(ctx, req, base_url=SURP)
    assert req["reasoning_effort"] == "medium"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__} {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")
    sys.exit(1 if failures else 0)
