# Research Agent Brief

Use this brief for the Gemini Research CLI.

## Mission

Find papers, technical reports, and GitHub repositories that can inform a CIM
chip for SNN on-chip learning, especially e-prop-like local learning and
hardware-feasible eligibility traces.

Research output should be useful even if most ideas are rejected. Do not modify
Python source code or source-of-truth spec files.

## NotebookLM Usage

Use the `notebooklm` skill as the research memory for candidate papers and
technical notes. The current policy is intake-first: when the Research Agent is
uncertain whether a paper belongs in the long-term vault, it should still place
the source into the NotebookLM vault, let NotebookLM generate a summary, and then
flag possible removal candidates for the user to decide.

Good fit:

- Broad intake of candidate PDFs/Markdown notes when relevance is uncertain.
- Deep-read selected PDFs after candidate papers are known.
- Maintain a local literature wiki with cited summaries.
- Track concepts such as eligibility trace storage, local learning signals,
  fixed-point e-prop, and CIM write-update constraints.
- Answer focused questions from already-synced sources.

Not a good fit:

- Broad first-pass web search for new papers or GitHub repositories.
- Quick one-off links that are unlikely to be used.
- Source code review of this repository.

Recommended flow:

1. Discover candidates with web/GitHub search or a literature-search workflow.
2. Add candidate PDFs/Markdown notes to the NotebookLM vault, even if relevance
   is uncertain.
3. Run the NotebookLM lifecycle command `init` or `update` from that vault.
4. Use generated summaries to mark each source as `keep`, `maybe_remove`, or
   `remove_candidate`.
5. Put removal recommendations in `docs/research/notebooklm-pruning.md` or the
   vault's `wiki/context/` area. Do not delete sources automatically.
6. Summarize only adopted/deferred findings into `docs/LITERATURE.md`.
7. Keep exploratory NotebookLM wiki content out of the main coding context.

Removal-recommendation template:

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

## IEEE PDF Download Skill

Use `$literature-search` as the PDF-download backend after IEEE candidate papers
are known. This is not the discovery step.

Scope:

- DOI lists.
- IEEE Xplore URLs.
- User-provided IEEE paper titles plus links.
- Download reports for known papers.

Rules:

- Use only legitimate IEEE access through the user's normal Chrome cookies,
  a dedicated Chrome CDP session, campus/institutional access, or Zotero
  Connector.
- Never bypass paywalls or access controls.
- If IEEE access is blocked, report `manual_required`.
- Download into the NotebookLM vault's `raw/papers/` when the source should be
  ingested, then run the NotebookLM `update` lifecycle command.
- Every item must receive a status: `downloaded`, `already_exists`,
  `manual_required`, or `failed_manual_required`.

## Topics To Search

- Bellec e-prop and follow-up implementations.
- ReckOn and other SNN processors with on-chip learning.
- CIM or compute-in-memory accelerators for SNNs.
- Local learning rules: e-prop, DECOLLE, surrogate local learning, STDP variants.
- Quantized eligibility traces and fixed-point SNN training.
- SRAM/RRAM/MRAM CIM write-update limits for on-chip training.
- Public GitHub repositories with runnable e-prop or SNN local-learning code.

## Required Output Location

Write structured results to:

- `docs/LITERATURE.md` for curated entries.
- `docs/research/YYYY-MM-DD-topic.md` for longer notes.
- `references/papers/` only for legally accessible PDFs or metadata files.
- `references/repos/` only for links, metadata, or small notes; do not vendor
  full external repositories unless explicitly requested.

## Entry Template

```text
## [Short title]

Source:
Type: paper / repo / benchmark / hardware chip
Link:
Key claim:
Evidence:
Hardware implication:
Conflict with current spec:
Possible local experiment:
Reproducibility:
License/code availability:
Risk:
Adopt / Reject / Defer:
```

## Rules

- Prefer primary sources: papers, official project repositories, and official
  documentation.
- Include publication year or repository activity date.
- Do not paste long excerpts.
- Separate paper claims from your own inference.
- Mark uncertain claims as uncertain.
- Always include "Conflict with current spec", even if the answer is "none
  known".
