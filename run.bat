@echo off
if not exist "env" (
    virtualenv env
    pip install -r requirements.txt
)

call .\env\Scripts\activate
python main.py
pause >nul



