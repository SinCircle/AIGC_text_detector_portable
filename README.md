## AIGC Text Detector (Portable + Streamlit UI)

简体中文 | English

本仓库在原项目的基础上，提供了一个基于 Streamlit 的本地可视化界面，并支持将模型打包到本地以便离线/便携使用。

原项目（算法与模型作者）：https://github.com/YuchuanTian/AIGC_text_detector

本仓库仅对“使用方式与交互界面”做了工程化改造，核心检测模型与算法完全沿用原项目/对应 Hugging Face 模型。

---

### 功能概述

- 交互界面：提供基于 Streamlit 的易用 Web 界面，支持中文/英文双模型、段落级检测、PDF/Word/Markdown/LaTeX 等常见文件导入与结果导出。
- 本地优先：若在 `models/` 目录下存在对应模型，将优先本地加载，适合无网或离线环境。
- 便携打包：一键将代码与模型打包为 ZIP，跨机器快速分发与使用。

---

### 目录结构

- 应用入口：[start.py](start.py)（自动检查并安装依赖，随后启动 Streamlit）
- Web 界面：[app_streamlit.py](app_streamlit.py)
- 检测引擎：[advanced_detector.py](advanced_detector.py)（封装本地优先加载与推理）
- 模型下载：[download_models.py](download_models.py)（将 Hugging Face 模型下载到本地 `models/`）
- 便携打包：[create_portable.py](create_portable.py)
- 本地模型目录：models/（例如 `AIGC_detector_zhv3`、`AIGC_detector_env3`）

---

### 快速开始（推荐）

1) 准备 Python 3.9+ 环境并安装依赖：

```powershell
cd "c:\Users\qyf06\Desktop\AIGC_text_detector_portable"
python -m pip install -r requirements.txt
```

2) 运行本地可视化界面：

```powershell
python start.py
```

或直接运行 Streamlit：

```powershell
streamlit run app_streamlit.py
```

打开浏览器访问 http://localhost:8501 即可开始使用。

---

### 获取模型（两种方式）

- 在线自动下载：首次运行时由 `transformers` 自动下载（需要网络）
- 本地离线加载：将模型放入本仓库的 `models/` 目录即可离线使用。

常用模型（由原项目提供）：
- 中文：`yuchuantian/AIGC_detector_zhv3`
- 英文：`yuchuantian/AIGC_detector_env3`

使用脚本下载到本地：

```powershell
python download_models.py
```

下载完成后，目录大致如下：

```
models/
  AIGC_detector_zhv3/
    config.json  tokenizer.json  model.safetensors ...
  AIGC_detector_env3/
    config.json  tokenizer.json  model.safetensors ...
```

---

### 便携打包

将当前工程和本地模型打包成一个 ZIP，方便在没有网络的环境快速分发：

```powershell
python create_portable.py
```

脚本会自动检查并包含 `models/` 目录（若不存在会尝试先下载）。

---

### 常见问题

- 首次运行较慢？模型会在第一次加载时初始化，之后会更快。
- GPU/CPU 切换？当前默认使用 CPU，可在 [advanced_detector.py](advanced_detector.py) 初始化时传入 `device="cuda"` 以启用 GPU（需本机已正确安装 CUDA 版 PyTorch）。
- 无网环境？提前执行 `python download_models.py` 将模型缓存到 `models/`，或把他人打包好的 `models/` 目录直接复制过来。

---

### 许可与致谢

- 本仓库尊重并遵循原项目的开源授权与署名要求，核心模型与算法归原作者所有。
- 原项目地址：https://github.com/YuchuanTian/AIGC_text_detector
- 模型来源：Hugging Face `yuchuantian/AIGC_detector_zhv3` 与 `yuchuantian/AIGC_detector_env3`

如本仓库对上游内容的引用存在表述不当之处，请在 Issue 中指出，我们会第一时间修正。
