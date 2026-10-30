@echo off
rem ============================================================
rem  Keymento gaze experiment tools  --  keep this file ASCII-only.
rem  cmd.exe parses .bat with the active codepage, so non-ASCII
rem  text here breaks parsing. Korean messages come from Python.
rem
rem  Independent of Keymento: its own venv (.venv),
rem  because mediapipe pulls in opencv-contrib-python, which would
rem  clash with the opencv-python that Keymento's AR window uses.
rem ============================================================
setlocal
cd /d "%~dp0"

chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PY=.venv\Scripts\python.exe

rem Collect everything after the command word so extra flags still
rem reach the script -- e.g. "run record --participant P01 --condition AR".
set "WORD=%~1"
shift
set "REST="
:collect
if "%~1"=="" goto :dispatch
set "REST=%REST% %1"
shift
goto :collect

:dispatch
if /i "%WORD%"=="setup" goto :setup
if /i "%WORD%"=="" goto :help
if /i "%WORD%"=="help" goto :help
if /i "%WORD%"=="/?" goto :help

if not exist "%PY%" (
    echo [!] venv not found: %CD%\%PY%
    echo     First run:  run setup
    exit /b 1
)

if /i "%WORD%"=="record"   goto :record
if /i "%WORD%"=="validate" goto :validate
if /i "%WORD%"=="analyze"  goto :analyze
if /i "%WORD%"=="test"     goto :test
echo [!] unknown command: %WORD%
goto :help

:setup
if not exist "%PY%" (
    python -m venv .venv || exit /b 1
)
"%PY%" -m pip install -r requirements.txt
goto :end

:record
"%PY%" gaze\record_face.py%REST%
goto :end

:validate
"%PY%" gaze\record_face.py --condition validation%REST%
goto :end

:analyze
"%PY%" gaze\analyze.py%REST%
goto :end

:test
"%PY%" -m unittest discover -s tests -p "test_*.py"
goto :end

:help
echo.
echo  Keymento gaze experiment  (run inside the keymento-gaze folder)
echo.
echo    run setup                                    first time: venv + mediapipe
echo    run record --participant P01 --condition AR  record face while Keymento runs
echo    run validate --participant P01               beep-cued validation (press V)
echo    run analyze [data\P01] [--review]            analyze recordings
echo    run test                                     unit tests
echo.
echo  Details: README.md
echo.

:end
endlocal
