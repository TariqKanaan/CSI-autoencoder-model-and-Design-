@echo off
REM ------------------------------------------------------------
REM limited_tree.bat
REM 
REM Recursively print a “tree /f”-style listing, but if a folder
REM is under "Data", only show up to 20 files in that folder.
REM Suppresses "File Not Found" by redirecting dir errors.
REM ------------------------------------------------------------

:: ─── 1) CONFIGURATION ────────────────────────────────────────
set "maxFiles=20"

:: If you pass a path as the first argument, use it; otherwise, use current directory
if "%~1"=="" (
    set "rootPath=%cd%"
) else (
    set "rootPath=%~1"
)

:: Compute the absolute path of the top‐level "Data" folder
set "dataFolder=%rootPath%\Data"

:: ─── 2) START RECURSION ──────────────────────────────────────
call :ListDir "%rootPath%" 0
goto :EOF

:: ─── 3) SUBROUTINE :ListDir ───────────────────────────────────
:ListDir
rem %~1 = full folder path, %~2 = depth level (integer)
setlocal EnableDelayedExpansion

set "currentFolder=%~1"
set "depth=%~2"

:: 3a) Build indentation string (e.g. depth=2 → "│   │   ")
set "spaces="
for /L %%i in (1,1,!depth!) do set "spaces=!spaces!│   "

:: 3b) Echo the folder name itself
echo !spaces!%~nx1

pushd "!currentFolder!" >nul 2>&1 || (
    rem If pushd fails (e.g. inaccessible), skip recursion
    exit /b
)

:: 3c) Determine if this folder is “under” Data
set "isUnderData=0"
for /f "delims=" %%X in ('cmd /v:on /c echo "!currentFolder!" ^| find /I "%dataFolder%"') do set "isUnderData=1"

:: ─── 3c(i) LIST FILES ─────────────────────────────────────────
set /a fileCount=0
if "!isUnderData!"=="1" (
    rem Only show up to maxFiles, suppress errors
    for /f "delims=" %%F in ('dir /b /a:-d 2^>nul') do (
        if !fileCount! lss %maxFiles% (
            echo !spaces!│   %%F
        )
        set /a fileCount+=1
    )
    if !fileCount! gtr %maxFiles% (
        set /a extra=fileCount - maxFiles
        echo !spaces!│   ... [!extra! more files omitted]
    )
) else (
    rem Show ALL files in this folder, suppress errors
    for /f "delims=" %%F in ('dir /b /a:-d 2^>nul') do (
        echo !spaces!│   %%F
    )
)

:: ─── 3c(ii) RECURSE INTO SUBFOLDERS ──────────────────────────
for /d %%D in (*) do (
    set /a nextDepth=depth+1
    call :ListDir "%%~fD" !nextDepth!
)

popd >nul
endlocal
exit /b
