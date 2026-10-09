# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: tray only. ASR stays as Python sources + embeddable runtime."""

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

pystray_datas, pystray_binaries, pystray_hidden = collect_all("pystray")
pil_datas, pil_binaries, pil_hidden = collect_all("PIL")

a = Analysis(
    ["../src/tray_app.py"],
    pathex=["../src"],
    binaries=pystray_binaries + pil_binaries,
    datas=pystray_datas + pil_datas,
    hiddenimports=(
        ["pystray", "pystray._win32", "pystray._base", "PIL", "PIL.Image", "PIL.ImageDraw",
         "win32com", "win32com.client", "pythoncom"]
        + collect_submodules("pystray")
        + pystray_hidden
        + pil_hidden
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ASRTray",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
