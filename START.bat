@echo off
setlocal

set "ROOT=%~dp0"

if exist "%ROOT%.venv\Scripts\python.exe" (
	start "AI Business Advisor API" "%ComSpec%" /k "cd /d ""%ROOT%backend"" && ""%ROOT%.venv\Scripts\python.exe"" -m uvicorn app:app --reload"
) else if exist "%ROOT%..\.venv\Scripts\python.exe" (
	start "AI Business Advisor API" "%ComSpec%" /k "cd /d ""%ROOT%backend"" && ""%ROOT%..\.venv\Scripts\python.exe"" -m uvicorn app:app --reload"
) else (
	start "AI Business Advisor API" "%ComSpec%" /k "cd /d ""%ROOT%backend"" && python -m uvicorn app:app --reload"
)

start "AI Business Advisor Frontend" "%ComSpec%" /k "cd /d ""%ROOT%frontend"" && npm run dev -- --host 127.0.0.1"

endlocal
