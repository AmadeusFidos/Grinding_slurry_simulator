@echo off
rem ---------------------------------------------------------------------------
rem  Build LapSim.exe, the desktop window, as a standalone Windows program.
rem
rem  Run it from the folder you cloned the repository into, either by
rem  double-clicking it or from a command prompt:
rem      build_exe.cmd
rem
rem  Needs Python 3.10 or newer from python.org. numpy, scipy, matplotlib and
rem  PyInstaller are installed into a private .venv-build folder here, so your
rem  main Python installation is left alone. The first run downloads about
rem  100 MB and takes a few minutes; later runs reuse the download.
rem
rem  Result: dist\LapSim\LapSim.exe. Keep it inside that folder -- the files
rem  beside it are the Python runtime it needs. Copy or zip the whole folder to
rem  move it or share it. Figures you render appear in dist\LapSim\figures.
rem ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

set "PY=py -3"
%PY% --version >nul 2>nul || set "PY=python"
%PY% --version >nul 2>nul || goto :no_python

if not exist ".venv-build\Scripts\python.exe" (
    echo Creating a private Python environment in .venv-build ...
    %PY% -m venv .venv-build || goto :failed
)
set "VPY=.venv-build\Scripts\python.exe"

echo Installing numpy, scipy, matplotlib and PyInstaller ...
"%VPY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt pyinstaller || goto :failed

echo Building LapSim.exe ...
"%VPY%" -m PyInstaller --noconfirm --clean --log-level WARN --windowed --name LapSim run_gui.py || goto :failed

rem Bring along the figures already rendered, so the Figures tab has something to show.
if exist figures\*.png xcopy /y /q /i figures\*.png dist\LapSim\figures >nul

echo Checking the finished program ...
start "" /wait "dist\LapSim\LapSim.exe" --check
if errorlevel 1 goto :check_failed
type "dist\LapSim\lapsim-check.log"
echo.
echo Done: %CD%\dist\LapSim\LapSim.exe
goto :end

:no_python
echo Python was not found. Install Python 3.10 or newer from
echo https://www.python.org/downloads/ and run this again.
set "FAILED=1"
goto :end

:check_failed
echo The program was built but failed its self-check:
type "dist\LapSim\lapsim-check.log"
set "FAILED=1"
goto :end

:failed
echo Build failed; see the messages above.
set "FAILED=1"

:end
rem Keep the window open when double-clicked; set LAPSIM_NO_PAUSE=1 to skip.
if not defined LAPSIM_NO_PAUSE pause
if defined FAILED exit /b 1
exit /b 0
