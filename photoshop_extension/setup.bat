@echo off
chcp 65001 >nul
title ISBD Studio AI - Photoshop Extension Installer
color 0A

echo =====================================================================
echo       ISBD Studio AI v1.00 - Photoshop Extension Auto-Installer
echo       (Adobe Photoshop CC 2018 / 2019 / 2020 / 2021 / 2022 / 2023 / 2024 / 2025 / 2026+)
echo =====================================================================
echo.

set SCRIPT_DIR=%~dp0
set EXT_SRC=%SCRIPT_DIR%extension
set TARGET_DIR=%APPDATA%\Adobe\CEP\extensions\com.isbd.studio.ai

if not exist "%EXT_SRC%" (
    color 0C
    echo [ত্রুটি] extension ফোল্ডারটি পাওয়া যায়নি!
    echo অনুগ্রহ করে জিপ ফাইলটি সম্পূর্ণ এক্সট্র্যাক্ট করে আবার চালান।
    pause
    exit /b 1
)

echo [1/3] অ্যাডোবি CEP রেজিস্ট্রি ডিবাগ মোড সক্রিয় করা হচ্ছে...
reg add "HKCU\Software\Adobe\CSXS.9" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.10" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.11" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.12" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.13" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.14" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.15" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
reg add "HKCU\Software\Adobe\CSXS.16" /v PlayerDebugMode /t REG_SZ /d "1" /f >nul 2>&1
echo       ✓ CEP PlayerDebugMode রেজিস্ট্রি কনফিগারেশন সম্পন্ন!
echo.

echo [2/3] ফটোশপ এক্সটেনশন ডিরেক্টরি প্রস্তুত করা হচ্ছে...
if not exist "%APPDATA%\Adobe\CEP\extensions" (
    mkdir "%APPDATA%\Adobe\CEP\extensions" >nul 2>&1
)

if exist "%TARGET_DIR%" (
    echo       * পূর্ববর্তী ভার্সন পাওয়া গেছে, আপডেট করা হচ্ছে...
    rmdir /s /q "%TARGET_DIR%" >nul 2>&1
)
mkdir "%TARGET_DIR%" >nul 2>&1
echo       ✓ টার্গেট ফোল্ডার তৈরি হয়েছে: %TARGET_DIR%
echo.

echo [3/3] এক্সটেনশন ফাইলসমূহ কপি করা হচ্ছে...
xcopy /s /e /y /q "%EXT_SRC%\*" "%TARGET_DIR%\" >nul
if %errorlevel% neq 0 (
    color 0C
    echo [ত্রুটি] ফাইল কপি করতে ব্যর্থ হয়েছে!
    pause
    exit /b 1
)
echo       ✓ সকল ফাইল সফলভাবে কপি সম্পন্ন হয়েছে!
echo.

echo =====================================================================
echo   ✅ অভিনন্দন! ISBD Studio AI ফটোশপ এক্সটেনশন সফলভাবে ইনস্টল হয়েছে!
echo.
echo   ব্যবহারবিধি:
echo   ১. Adobe Photoshop চালু করুন।
echo   ২. মেনু থেকে যান: Window ➔ Extensions ➔ ISBD Studio AI
echo   ৩. আপনার ক্যানভাসে যেকোনো ছবি খুলে ওয়ান-ক্লিকে রিটাচ ও ক্লিপিং করুন!
echo =====================================================================
echo.
pause
