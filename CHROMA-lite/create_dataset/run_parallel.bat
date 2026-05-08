@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Use the folder where this BAT lives as the working dir
set "ROOT=%~dp0"
pushd "%ROOT%"
mkdir "logs" 2>nul

REM Adjust concurrency
set "MAX_JOBS=10"

REM ---------------- MAIN WORK ----------------
for %%L in (4 6 8) do (
  for %%A in (0 30 60) do (
    for /l %%S in (42,42,4200) do (
      call :wait_for_slot %MAX_JOBS%
      echo [LAUNCH] L=%%L A=%%A S=%%S
      start "" /b /d "%ROOT%" cmd /c ^
        python -u src\show_datasets.py --num_layers %%L --incidence_angle %%A --seed %%S ^
        1>"%ROOT%logs\L%%L_A%%A_S%%S.out" 2>"%ROOT%logs\L%%L_A%%A_S%%S.err"
    )
  )
)

call :wait_for_all
echo [DONE] All workers finished.
popd
exit /b 0

REM ---------------- HELPERS ----------------
:wait_for_slot
set "LIMIT=%~1"
:check_slot
for /f %%C in ('tasklist /fi "imagename eq python.exe" ^| find /i /c "python.exe"') do set COUNT=%%C
if !COUNT! GEQ !LIMIT! (
  >nul ping 127.0.0.1 -n 3
  goto :check_slot
)
exit /b 0

:wait_for_all
for /f %%C in ('tasklist /fi "imagename eq python.exe" ^| find /i /c "python.exe"') do set COUNT=%%C
if !COUNT! GTR 0 (
  >nul ping 127.0.0.1 -n 3
  goto :wait_for_all
)
exit /b 0
