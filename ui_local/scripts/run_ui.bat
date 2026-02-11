@echo off
cd /d %~dp0\..
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8787 --reload
