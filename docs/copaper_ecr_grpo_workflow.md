# CoPaper Workflow For ECR-GRPO

## Deployment State

CoPaper source code is outside this repository at:

```text
E:\yf\CoPaper-OpenCode
```

The ECR-GRPO project keeps only project-specific CoPaper files:

```text
.agents/
.opencode/
storyline.md
paper.md
writingrules.md
AGENTS.md
templates/
opencode.json
.env.example
```

Local deployment artifacts are ignored by git:

```text
node_modules/
.venv/
.venv-copaper/
third_party/
.env
.env.local
```

## What Opens CoPaper?

CoPaper is not a standalone GUI application.

Use VS Code to edit `storyline.md`, `paper.md`, and experiment notes. Use the terminal for `copaper` CLI commands. If you use OpenCode, the `.opencode/commands/` files provide slash commands such as `/copaper`, `/copaper-doctor`, and `/copaper-relatedwork`.

Practical workflow:

```text
VS Code = edit and inspect paper files
Terminal = run copaper CLI
OpenCode = use CoPaper slash commands and agents
```

## API And Relay Configuration

CoPaper related-work tools read these environment variables from `.env`:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=...       # optional OpenAI-compatible relay/proxy endpoint
COPAPER_MODEL=...
```

Do not commit `.env`. Use `.env.example` as the template.

OpenCode provider credentials are managed by OpenCode itself. CoPaper's generated `.opencode` folder contains command/plugin wiring, not secret keys.

## Commands

Check CoPaper Python state:

```powershell
.\.venv-copaper\Scripts\copaper.exe --root . status
```

Check OpenCode integration:

```powershell
$env:Path = 'C:\Users\12875\AppData\Local\Microsoft\WinGet\Packages\Oven-sh.Bun_Microsoft.Winget.Source_8wekyb3d8bbwe\bun-windows-x64;' + $env:Path
.\node_modules\.bin\copaper-opencode.exe doctor --format json
```

After opening this repository in OpenCode, restart OpenCode and run:

```text
/copaper-doctor
/copaper
```

## Paper Plan

Use `storyline.md` as the research narrative and `paper.md` as the manuscript skeleton.

Recommended phase order:

1. Storyline: confirm problem, insight, contribution, and claim boundary.
2. Literature: collect GRPO/PPO agentic RL, step-level optimization, delayed reward credit assignment, and tool-use RL references.
3. Experiments: convert existing `runs/` outputs into publication tables and figures.
4. Writing: fill `paper.md` section by section.
5. Review: run CoPaper checkers and revise.
6. LaTeX: export only after Markdown structure is stable.

## Immediate Work Items

1. Decide whether the first paper claims only controlled synthetic + HF/LoRA evidence, or also includes real benchmark evidence.
2. Generate paper-ready CSV summaries and plots from existing `runs/`.
3. Pick one credit-assignment case study from `credit_assignments.jsonl`.
4. Fill the related-work bibliography.
5. Write Introduction and Method first; defer abstract polish until results tables are fixed.
