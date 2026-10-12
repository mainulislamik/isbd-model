@echo off
chcp 65001 >nul
title ISBD Studio AI - Extension Uninstaller
color 0E

echo =====================================================================
echo       ISBD Studio AI - Photoshop Extension আন-ইনস্টলার
echo =====================================================================
echo.

set TARGET_DIR=%APPDATA%\Adobe\CEP\extensions\com.isbd.studio.ai

if exist "%TARGET_DIR%" (
    echo এক্সটেনশন ফোল্ডারটি মুছে ফেলা হচ্ছে: %TARGET_DIR%
    rmdir /s /q "%TARGET_DIR%" >nul 2>&1
    echo.
    echo ✅ ISBD Studio AI এক্সটেনশন সফলভাবে অপসারিত হয়েছে!
) else (
    echo এক্সটেনশনটি ইনস্টল করা নেই।
)

echo.
pause
