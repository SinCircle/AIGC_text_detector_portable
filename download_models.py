"""
下载并保存 HuggingFace 上的检测模型到本地 `models/` 目录，便于制作携带版

用法:
    python download_models.py

该脚本会把模型保存到 `./models/<模型名>/` 下，例如 `models/AIGC_detector_zhv3`。
"""
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODELS = [
    "yuchuantian/AIGC_detector_zhv3",
    "yuchuantian/AIGC_detector_env3"
]


def download_and_save(model_id: str, root: Path):
    name = model_id.split('/')[-1]
    local_dir = root / name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} -> {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"Saved to {local_dir}\n")


def main():
    root = Path(__file__).parent.resolve() / "models"
    root.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        try:
            download_and_save(m, root)
        except Exception as e:
            print(f"下载模型 {m} 失败: {e}")


if __name__ == '__main__':
    main()
