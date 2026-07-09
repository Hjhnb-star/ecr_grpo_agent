# ECR-GRPO LaTeX Readiness Report

## Status

`latex_review` cannot be completed as a final template export yet because `templates/latex/` does not contain a `.tex` conference or journal template. The directory currently contains only `templates/latex/README.md`.

This is a workflow blocker, not a writing blocker: `paper.md`, `relatedwork/paper_list.bib`, and the review reports are ready to be used once a target template is provided.

## Current Inputs

| Input | Status | Notes |
|---|---|---|
| `paper.md` | present | Framework draft with claim-boundary fixes applied. |
| `storyline.md` | present | Coherent storyline and experiment ladder. |
| `relatedwork/paper_list.bib` | present | 13 seed references generated from downloaded PDFs. |
| `relatedwork/papers/*.md` | present | 13 LLM-generated paper summaries. |
| `docs/ecr_grpo_literature_matrix.md` | present | Human-readable related-work positioning. |
| `docs/ecr_grpo_experiment_evidence_report.md` | present | Evidence/source audit. |
| `docs/ecr_grpo_paper_review_report.md` | present | Seven-checker static review. |
| `fig/` | not checked in this pass | Publication figures still need to be generated or copied. |
| `templates/latex/*.tex` | missing | Required before final LaTeX export. |

## Required Template Files

Place the target venue template under `templates/latex/`. For AAAI-style export, this usually means a directory containing files such as:

- a main sample `.tex` file
- required `.sty` or `.cls` files
- bibliography style files if provided
- any template-specific assets

Do not place generated output in `templates/latex/`. The export target should be `target/latex/`.

## Export Plan Once Template Exists

1. Identify the main `.tex` template file.
2. Copy the template tree into `target/latex/`.
3. Convert `paper.md` sections into LaTeX.
4. Copy `relatedwork/paper_list.bib` into `target/latex/`.
5. Wire the bibliography command to the copied `.bib` file.
6. Insert placeholders for figures/tables whose final assets are not ready.
7. Compile if a LaTeX engine is installed.

## Blocking Items Before Final Submission

1. Restore/regenerate source tables for Stage A and Stage B.
2. Complete or clearly remove Stage C/ALFWorld result claims.
3. Generate publication-ready figures.
4. Add reproduction commands under every table.
5. Provide the final venue template.

## Recommended Next Command After Adding Template

After placing the venue template into `templates/latex/`, rerun the LaTeX export workflow from the project root:

```powershell
.\.venv-copaper\Scripts\copaper.exe --root . status
```

Then ask Codex to export `paper.md` using the template in `templates/latex/`.
