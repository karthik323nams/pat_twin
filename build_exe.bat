@echo off
REM Run on Windows with Python 3.10-3.12. Produces dist\PAT-Twin\PAT-Twin.exe
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name PAT-Twin --collect-all cv2 pat_gui.py
echo Test dist\PAT-Twin\PAT-Twin.exe on a machine WITHOUT Python.
