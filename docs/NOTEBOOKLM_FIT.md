# NotebookLM Skill Fit

Date: 2026-05-11
Owner: Main Agent / Codex

## Verdict

The `notebooklm` skill fits this project as a research knowledge-base layer, not
as the first-pass discovery agent.

Use it when Gemini or another research CLI has identified candidate papers,
technical notes, or Markdown summaries. The policy for this project is
intake-first: if relevance is uncertain before reading, put the source into the
NotebookLM vault, generate a summary, and then ask Gemini to flag sources that
may not belong in the vault. The user makes the final removal decision.

## Why It Fits

Current project needs:

- e-prop paper lineage.
- SNN on-chip learning papers.
- CIM/SRAM/RRAM/MRAM write-update constraints.
- Hardware feasibility notes for eligibility traces and learning signals.
- A way to preserve useful research without polluting the Main Agent context.

The skill provides:

- `raw/` for original PDFs/Markdown.
- `wiki/sources/` for NotebookLM-generated source summaries.
- `wiki/concepts/` for concept extraction.
- `wiki/index.md` for literature index.
- `wiki/context/` for cross-tool research memory.
- Citation-preserving summaries with markers such as `[1]`.

## Limits

The skill is not enough by itself for the full Research Agent role:

- It does not discover papers or GitHub repositories by itself.
- It should not be used for quick broad web search.
- It should not be used to review local Python source code.
- It requires a vault root and NotebookLM access/login/proxy to work.

## Recommended Project Setup

Create a separate research vault outside the main code path, for example:

```text
references/notebooklm-vault/
  raw/
  wiki/
  .notebooklm-vault.json
```

Then run lifecycle commands from the vault root:

```bash
/Users/lixiangting/.cc-switch/skills/notebooklm/notebooklm.sh init
/Users/lixiangting/.cc-switch/skills/notebooklm/notebooklm.sh update
/Users/lixiangting/.cc-switch/skills/notebooklm/notebooklm.sh status
```

Do not pipe these commands through `head`, `tail`, or `sed`; the skill explicitly
requires full streaming output.

## Recommended Research Flow

1. Use Gemini/web/literature-search for broad discovery.
2. Put candidate PDFs or Markdown notes into the NotebookLM vault, including
   uncertain ones.
3. Run `init` or `update`.
4. Let the Research Agent maintain detailed summaries inside the vault.
5. Ask the Research Agent to produce a pruning list from the summaries:
   `keep`, `maybe_remove`, or `remove_candidate`.
6. Do not delete sources automatically; wait for the user's decision.
7. Copy only concise adopted/deferred implications into `docs/LITERATURE.md`.
8. Main Agent reads `docs/LITERATURE.md` first, and opens the vault only for
   specific decisions.

## Pruning Report Template

```text
Source:
Why it entered the vault:
Summary-based relevance:
Reason to keep:
Reason it may not belong:
Risk if removed:
Recommendation: keep / maybe_remove / remove_candidate
User decision:
```

## Fit For Current Agent Split

This skill is a good backend for Gemini Research:

- Gemini can use it to organize and deep-read accepted sources.
- Codex Main should not ingest the whole vault by default.
- Claude Review should use only curated findings unless asked to audit research
  claims.
