$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.trainer --config configs\smoke.json
python -m ecr_grpo.run_baselines --config configs\smoke.json --updates 20
python -m unittest discover tests
