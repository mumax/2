@echo off

set MUMAX2=%~dp0\..
echo MUMAXPATH: %MUMAX2%
set PYTHONPATH=%PYTHONPATH%;%MUMAX2%\src\python
set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%;%MUMAX2%\src\libmumax

echo %PYTHONPATH%

%MUMAX2%\bin\mumax2-bin.exe %*

