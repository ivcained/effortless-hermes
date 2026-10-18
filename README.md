<img width="1200" height="600" alt="effortless-hermes" src="https://github.com/user-attachments/assets/ac690ccc-148b-43d6-9232-9abc0447e991" />


# effortless-hermes

A Hermes port of [HeyCubit/effortless](https://github.com/HeyCubit/effortless) (MIT): a fast
judge reads each prompt once per turn and picks the reasoning effort for it — easy questions
run cheap, hard jobs get high — by rewriting the LLM request before it ships.

## How it works

- An `llm_request` middleware fires once per user turn (keyed on `turn_id`), reads the next
  user message plus recent context, asks a judge, and rewrites every reasoning-effort
  carrier already present in the provider kwargs (`reasoning_effort`, `extra_body.reasoning.effort`,
  `reasoning.effort`). It never invents a carrier a provider doesn't use, never touches
  `messages`, and never changes the model — so the prompt-cache prefix stays intact.
- On a retried request within the same turn the cached verdict is reapplied.
- A named level in the message ("max effort", "effort: high") wins without a judge; a short
  follow-up ("go", "ok") keeps the current effort; a judge that is unsure (<0.5) or down
  falls through to the fallback judge, and if that fails too the current effort is kept.
- Verdicts are logged (last 50) in profile-scoped plugin state and shown by `/effortless`
  in interactive sessions.

## Judges

| Backend | Config | Notes |
|---|---|---|
| `jev` (default) | `typesafe_key` setting (stored as `EFFORTLESS_HERMES_TYPESAFE_KEY` in `.env`), or the shared `TYPESAFE_API_KEY` env var | TypeSafe systemone, ~0.3 s, returns confidence |
| `local` (default fallback) | `local_base_url` (default `http://127.0.0.1:8080/v1`), `local_model` | any OpenAI-compatible server; must be running |
| `custom` | `custom_base_url`, `custom_model`, `custom_api_key` (stored as `EFFORTLESS_HERMES_CUSTOM_API_KEY` in `.env`) | any OpenAI-compatible API |

Secrets are read only through Hermes' profile-scoped secret store (`agent.secret_scope`),
never from `.env` files or `os.environ` directly. When no key is set, the jev judge is
skipped and the fallback chain runs; when nothing answers, the current effort is kept —
the plugin never blocks a prompt.

## Settings (`plugins.entries.effortless-hermes.settings` in config.yaml)

`enabled` (bool, true) · `judge_backend` (jev|local|custom) · `fallback_backend` (local|custom|jev|none)
· `judge_timeout_seconds` (3.0) · `effort_floor` (low) · `effort_ceiling` (max) ·
`effort_bias` (-2..+2) · `model_map` (dict, off) · `model_map_always` (bool, false) ·
judge connection keys per the table above.

## Per-turn telemetry and `/effortless stats`

A `post_api_request` observer hook records the first request of every turn — input
tokens, context length, duration, model — into the plugin's profile-scoped state.
`/effortless` shows the latest context fill; `/effortless stats` aggregates the recent
window: verdicts per effort, per judge, average judge latency, average input tokens and
context fill. There is deliberately **no cost-savings estimate** — what a prompt would
have cost at another effort is not knowable (same honesty rule as upstream effortless).
Cache warmth is not reported: the hook payload carries usage and timing, but no explicit
cache-hit field, and we won't guess one.

## Effort→model mapping (opt-in, cache-aware)

`model_map` lets a verdict steer the model, not just the effort, for multi-model
gateways:

```yaml
plugins:
  entries:
    effortless-hermes:
      settings:
        model_map: { low: <cheap model id>, medium: <balanced id>, high: <strong id> }
```

- Applies to the whole turn (reapplied on retries), never mid-turn.
- **By default it only fires on a fresh chat** (first user turn). In an established
  chat, switching models re-reads the entire conversation at full price on the new
  model — the provider prompt cache is keyed by model. `model_map_always: true`
  opts into that cost explicitly.
- Model ids must be servable by the chat's **own provider** — the middleware swaps the
  model id on the request; it does not switch providers, keys, or endpoints.
- This is generic on purpose: any OpenAI-compatible router that serves several models
  behind one endpoint (a self-hosted router, an aggregator) can be driven by effort
  verdicts. No specific gateway is named or required.

## Commands

- `/effortless` — show the last verdicts
- `/effortless test` — send a sample prompt through both judges

## Security disclosure

- **Network calls**: with the default config the plugin sends your next message (up to
  4000 chars) and a short excerpt of recent conversation (up to 1600 chars) to
  `https://api.typesafe.ai/v1/systemone` (TypeSafe Jev) on the first request of each turn,
  solely to pick the effort. The `local` backend sends the same text to the base URL you
  configure (default `127.0.0.1:8080`); the `custom` backend to your configured API.
  No other data leaves the machine; there is no telemetry, no usage reporting, no file
  access beyond the plugin's own state.
- **Read-only telemetry**: a `post_api_request` observer hook reads token usage and
  context length from Hermes' own hook payloads and stores them locally in plugin state
  for the `/effortless stats` command. Nothing is sent anywhere.
- **Latency budget**: each judge is bounded by `judge_timeout_seconds` (default 3 s; local
  and custom judges allow up to 8 s). A judge that hangs is abandoned, not awaited.
- **Credentials**: keys come from the settings hub (stored in Hermes' `.env`) or the
  `TYPESAFE_API_KEY` env var, resolved through `agent.secret_scope.get_secret` so
  profile-scoped secrets stay scoped. The plugin never reads `.env` files itself and never
  writes credentials anywhere.
- **Code**: pure-stdlib Python (urllib), no pip dependencies, no shell-outs, no background
  processes, no self-updating code.

## Provider effort vocabularies (why your request might have 400'd)

Providers accept different `reasoning_effort` vocabularies, and some accept
different ones per route. The plugin clamps every effort it writes — and now
also every effort already on the request — to the vocabulary the current
provider/model pair accepts:

| Route | Accepted | medium maps to |
|---|---|---|
| Surplus Intelligence (`api.surplusintelligence.ai`), agentic requests (any model) | `low`, `high`, `max` | `high` |
| glm-5.3 (direct), glm-4.5, deepseek | `low`–`high`/`max` incl. `medium` | `medium` (legal) |
| kimi-k3 | `low`, `high`, `max` | `high` |
| unknown | full ladder | `medium` (legal) |

Two behaviors worth knowing:

- **Sanitize-before-ship**: if the request already carries an effort the
  provider would reject (a config-level `agent.reasoning_effort: medium` on a
  low/high/max route, for example), the middleware rewrites it to the nearest
  accepted level on *every* request — including follow-ups, judge-outage turns
  and retries — so a provider 400 (`'reasoning_effort' must be one of:
  'low', 'high', 'max'`) can't ship.
- **medium → high, not low**: on restricted ladders a judged `medium` maps up
  to `high`. The judge called the prompt "normal work: a few tool calls";
  `low` means "no tools, no planning", which would under-power it.

## Verified

- 2026-10-14, Hermes v0.21.7, glm-5.3 via surplusintelligence:
  "What day comes after Tuesday" → `reasoning_effort: low` in the live request dump
  (jev, 99% sure, 331 ms); "Refactor the entire auth system…" → `high` (82% sure);
  retry reapply confirmed via double dumps; `hermes plugins validate --install-deps` clean.

## Not in v0 (deliberate)

Per-prompt model routing (effortless keeps it off by default too), the desktop status bar
UI, the cache-countdown timer, handoff/compact triggers, and Anthropic `thinking` budget
carriers (providers without a text effort carrier are skipped, never guessed).

## License

MIT. Judge prompts and heuristics adapted from HeyCubit/effortless (MIT).

## Changelog

- 0.3.1 (2026-10-17): Surp-hosted glm-5.3 clamps to low/high/max. Live-verified:
  api.surplusintelligence.ai accepts medium/xhigh on plain chat but 400s on
  agentic requests (tools attached or long agent-style history) with
  "'reasoning_effort' must be one of: 'low', 'high', 'max'". Wire vocabularies
  now take `base_url`/`tools`, and cached per-turn efforts are re-clamped on
  retry re-apply instead of being pasted verbatim.

- 0.4.0 (2026-10-18): outcome feedback loop. A quality-class failure (reason
  `incomplete_response`/`empty_response`, or `finish_reason=error`) escalates
  that session's effort one step for the next turn AND for retries of the
  failed turn (retries now ship more thinking, not the verdict that just
  failed). Bias caps at +2; two clean first-calls decay it one step. Rate
  limit, auth, billing and request-shape 400s never escalate — those are not
  thinking problems. New `api_request_error` observer registration, new
  `feedback_enabled` setting (default true), `failure bias` line in
  `/effortless stats`, `finish_reason` recorded per turn, and the per-session
  maps are now bounded (200 sessions) so long-lived gateways don't leak.
