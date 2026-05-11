# Design Decisions

Status: active
Owner: Main Agent / Codex

Use this file to record decisions that should not be repeatedly debated.

## D-0001: Use A Main Agent With Isolated Research And Review Agents

Date: 2026-05-11
Status: accepted

Decision:

Use Codex as the Main Agent, Gemini as the Research Agent, and Claude as the
Review Agent. Add a Runner Agent only when experiments become long-running.

Rationale:

The repository is still in research-code cleanup. Too many always-on agents
would increase coordination overhead before there is a stable spec, run format,
or test framework.

Consequences:

- Main Agent owns source-of-truth docs and final code changes.
- Research is isolated to structured summaries and references.
- Review is read-only by default.
- Runner is optional and writes only experiment artifacts.

## D-0002: Keep Research Context Out Of The Main Thread

Date: 2026-05-11
Status: accepted

Decision:

Research output must be written as structured summaries under
`docs/LITERATURE.md` or `docs/research/`. Main Agent should read only the
entries needed for a current decision.

Rationale:

Most research findings may be rejected or deferred. Keeping them outside the
main coding context reduces accidental adoption of speculative ideas.

Consequences:

- Research entries must include conflict-with-spec and possible-experiment
  fields.
- Main Agent must explicitly adopt, reject, or defer significant ideas.

## D-0003: Rename New Speech E-prop Line To `eprop/`

Date: 2026-05-11
Status: accepted

Decision:

Keep the newer speech-recognition e-prop implementation as `eprop/`. The older
`eprop/` directory was removed from the workspace and temporarily backed up at
`/private/tmp/SNN_old_eprop_20260511-171229`. Cue accumulation is not part of
the target workflow and should not drive cleanup priorities.

Rationale:

The user confirmed cue accumulation is not needed. The current `eprop/`
directory contains the newer speech-specific and circuit-oriented work:
label-aware preprocessing,
`label.npy`-based speech targets, device-aware quantization assignment,
circuit-style LIF/reset options, output-spike state, and a saved checkpoint.

Consequences:

- First engineering tasks should make `eprop/model/main.py` run to
  completion on a tiny CPU speech run.
- `eprop` still needs cleanup before it is reliable. Detailed findings are
  recorded in `docs/EPROP_REVIEW.md`.
