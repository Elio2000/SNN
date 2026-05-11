# IEEE PDF Download Skill Fit

Date: 2026-05-11
Owner: Main Agent / Codex

## Verdict

The local PDF-download workflow is covered by the `literature-search` skill. I
did not find a separate skill named `pdk`; the available matching skill is:

```text
/Users/lixiangting/.cc-switch/skills/literature-search/SKILL.md
```

It is suitable for IEEE paper downloads after candidate papers are known. It is
not the primary discovery engine.

## What It Should Do

Use it for:

- IEEE Xplore URLs.
- IEEE DOI lists.
- User-provided IEEE paper titles plus DOI/URL.
- Downloading PDFs into a NotebookLM vault's `raw/papers/`.
- Reporting which papers still require manual download.

## What It Should Not Do

Do not use it for:

- Broad literature discovery.
- Bypassing IEEE paywalls or access controls.
- Repeatedly retrying blocked downloads without surfacing the failure.
- Source-code review.

## IEEE Access Rules

IEEE downloads are best-effort and must use legitimate access:

- Chrome cookies from a logged-in session.
- Dedicated Chrome CDP profile logged into IEEE through an institution.
- User's normal Chrome browser mode when it can download IEEE PDFs.
- Zotero Connector fallback when browser download is more reliable.

If access is blocked, the output should mark the paper as `manual_required` or
`failed_manual_required`.

## How It Connects To NotebookLM

Recommended flow:

1. Gemini finds candidate IEEE papers.
2. The download workflow fetches known IEEE PDFs into `raw/papers/`.
3. Run the NotebookLM vault `update` command.
4. Gemini reviews NotebookLM-generated summaries.
5. Gemini writes a pruning recommendation for sources that may not belong.
6. User decides whether to remove them from the vault.

