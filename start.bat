@echo off
chcp 936 >nul
title 药板药片外观缺陷智能检测系统 - 一键启动
cd /d "%~dp0"

echo ============================================================
echo   药板药片外观缺陷智能检测系统
echo   一键启动（后端服务 + 自动打开浏览器）
echo ============================================================
echo.

rem ---------- 1. 在同目录 / PATH 中查找 Python ----------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
    where py >nul 2>nul && set "PY=py"
)
if not defined PY (
    echo [错误] 未找到 Python。请安装 Python 3.9+ 并勾选「Add to PATH」。
    pause
    exit /b 1
)
echo [1/4] 使用 Python：%PY%

rem ---------- 2. 检查并安装依赖 ----------
%PY% -c "import fastapi, uvicorn, cv2, ultralytics, sklearn, numpy" >nul 2>nul
if errorlevel 1 (
    echo [2/4] 检测到缺少依赖，正在安装 requirements.txt ...
    %PY% -m pip install -r requirements.txt
) else (
    echo [2/4] 依赖已就绪
)

rem ---------- 3. 检查模型文件 ----------
echo [3/4] 检查模型文件 ...
if not exist "models\best.pt" echo   [警告] 缺少 models\best.pt，请运行 python scripts/train_yolo.py
if not exist "models\rf_classifier.pkl" echo   [警告] 缺少 models\rf_classifier.pkl，请运行 python scripts/train_rf.py

rem ---------- 4. 启动后端服务（新窗口）并打开浏览器 ----------
echo [4/4] 启动后端服务，稍后自动打开浏览器 ...
start "药板缺陷检测 - 后端服务" cmd /k "%PY% -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:8000/

echo.
echo ============================================================
echo   系统已启动！访问 http://127.0.0.1:8000/
echo   停止方法：关闭「药板缺陷检测 - 后端服务」窗口
echo   若页面未自动打开，请手动在浏览器输入上方地址
echo ============================================================
echo.
pause
