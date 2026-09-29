@echo off
rem ---------------------------------------------------------------------------
rem  Build (if needed) and launch the Windows Forms front end.
rem
rem  Run this from Windows, not from inside WSL. The project may live on a WSL
rem  path such as \\wsl.localhost\Ubuntu\... , which cmd cannot use as a working
rem  directory -- so nothing here depends on the current directory.
rem
rem  Set LAPSIM_PYTHON first if you want a particular interpreter, e.g.
rem      set LAPSIM_PYTHON=C:\Python313\python.exe
rem  It needs numpy and scipy. matplotlib is optional: without it everything
rem  works except rendering new figures.
rem ---------------------------------------------------------------------------

setlocal
set "PROJECT=%~dp0LapSim.UI\LapSim.UI.csproj"
set "EXE=%~dp0LapSim.UI\bin\Debug\net10.0-windows\LapSim.exe"

if not exist "%EXE%" (
    echo Building the front end...
    dotnet build "%PROJECT%" -v q --nologo
    if errorlevel 1 (
        echo.
        echo Build failed. You need the .NET SDK: https://dotnet.microsoft.com/download
        pause
        exit /b 1
    )
)

start "" "%EXE%"
endlocal
