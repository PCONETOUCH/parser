@echo off
setlocal
REM Compatibility wrapper: always use auto mode.
call "%~dp0run_unikey_parser_auto.bat" %*
endlocal
