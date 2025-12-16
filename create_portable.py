"""
创建项目的携带版 ZIP 包。

用法:
    python create_portable.py

注意：如果本地 models/ 目录不存在，脚本会尝试调用 `download_models.py` 先下载模型。
"""
import os
import shutil
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parent.resolve()
OUT_NAME = "AIGC_text_detector_portable"


def ensure_models():
    models_dir = ROOT / "models"
    if not models_dir.exists() or not any(models_dir.iterdir()):
        print("未检测到本地 models/ 目录，尝试下载模型（这可能需要较长时间）...")
        subprocess.check_call([sys.executable, str(ROOT / "download_models.py")])
    else:
        print("检测到本地 models/ 目录，跳过下载。")


def make_zip():
    bundle_dir = ROOT / (OUT_NAME + "_temp")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir()

    # 需要包含的文件/目录
    include = [
        "advanced_detector.py",
        "app_streamlit.py",
        "运行Web界面.py",
        "README.md",
        "requirements.txt",
        "download_models.py",
        "create_portable.py",
        "models"
    ]

    for name in include:
        src = ROOT / name
        if not src.exists():
            print(f"警告：缺少 {name}，跳过")
            continue

        dst = bundle_dir / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # 创建 zip
    out_path = ROOT / (OUT_NAME)
    if out_path.exists():
        if out_path.with_suffix('.zip').exists():
            out_path.with_suffix('.zip').unlink()

    shutil.make_archive(str(out_path), 'zip', root_dir=bundle_dir)
    shutil.rmtree(bundle_dir)

    print(f"已生成便携包: {out_path.with_suffix('.zip')}")


def main():
    ensure_models()
    make_zip()


if __name__ == '__main__':
    main()
