$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
py -3 -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -e '.[test]'
& .venv\Scripts\python.exe -m encino_waves doctor
