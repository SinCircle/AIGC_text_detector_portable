"""
Streamlit 前端 - 段落级别 AIGC 检测工具
支持逐段落检测和可视化显示
"""

import streamlit as st
import pandas as pd
from advanced_detector import ChineseAIGCDetector
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import re
import html
import PyPDF2
from docx import Document
import io
import jieba
import numpy as np
from collections import defaultdict

# 页面配置
st.set_page_config(
    page_title="AIGC 检测器",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
    <style>
    /* 紧凑式段落样式 - 类似原文布局 */
    .compact-text-container {
        font-size: 16px;
        line-height: 1.8;
        margin: 20px 0;
        padding: 0;
    }
    .text-segment {
        display: inline;
        padding: 2px 0;
        line-height: 1.8;
        transition: all 0.2s ease;
    }
    .text-segment:hover {
        opacity: 0.85;
        cursor: pointer;
    }
    .char-segment {
        display: inline;
        padding: 2px 0;
        line-height: 1.8;
        transition: all 0.2s ease;
        position: relative;
    }
    .char-segment:hover {
        opacity: 0.85;
        cursor: pointer;
    }
    /* AI 率高 - 红色背景 */
    .highlight-high {
        background-color: rgba(255, 100, 100, 0.25);
        border-bottom: 2px solid #ff6464;
    }
    /* AI 率中 - 黄色背景 */
    .highlight-medium {
        background-color: rgba(255, 193, 7, 0.25);
        border-bottom: 2px solid #ffc107;
    }
    /* AI 率低 - 绿色背景 */
    .highlight-low {
        background-color: rgba(76, 175, 80, 0.2);
        border-bottom: 2px solid #4caf50;
    }
    /* 内联标记 */
    .inline-badge {
        display: inline-block;
        font-size: 11px;
        padding: 2px 6px;
        margin: 0 3px;
        border-radius: 3px;
        font-weight: bold;
        vertical-align: super;
        line-height: 1;
    }
    .inline-badge-high {
        background-color: #ff6464;
        color: white;
    }
    .inline-badge-medium {
        background-color: #ffc107;
        color: #333;
    }
    .inline-badge-low {
        background-color: #4caf50;
        color: white;
    }
    /* 图例 */
    .legend-container {
        display: flex;
        gap: 20px;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 14px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .legend-color {
        width: 40px;
        height: 20px;
        border-radius: 3px;
        border: 1px solid #ddd;
    }
    /* 细小的内联编辑按钮：笔图标，极小尺寸 */
    .stTooltipHoverTarget button {
        border: none !important;
        background: transparent !important;
        padding: 0 3px !important;
        min-width: 16px !important;
        min-height: 14px !important;
        height: 16px !important;
        font-size: 12px !important;
        line-height: 1 !important;
        box-shadow: none !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dialog support check
if hasattr(st, "dialog"):
    dialog_decorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    dialog_decorator = st.experimental_dialog
else:
    dialog_decorator = None

def edit_form_content(index, detector: ChineseAIGCDetector = None):
    """编辑表单内容"""
    if "results" not in st.session_state or index >= len(st.session_state.results):
        st.error("数据错误")
        return

    item = st.session_state.results[index]
    active_detector = detector or st.session_state.get("detector")
    
    # 临时结果 key (用于重算但不提交的情况)
    temp_result_key = f"temp_result_{index}"
    display_item = st.session_state.get(temp_result_key, item)

    # 若上次点击了重置，清理输入框状态，再渲染新的默认值
    reset_flag_key = f"reset_request_{index}"
    if st.session_state.get(reset_flag_key):
        st.session_state.pop(reset_flag_key, None)
        st.session_state.pop(f"edit_area_{index}", None)
        st.session_state.pop(temp_result_key, None) # 清除临时结果
        display_item = item # 回退到原始结果
    
    metric_slot = st.empty()
    render_ai_metric(metric_slot, display_item["AI率"])
    
    # Original Text (Always Visible)
    st.text_area("原文显示", value=item.get("原文", item["文本"]), disabled=True, height=100)
    
    # Contribution View (Inserted between Original and Current if calculated)
    contrib_key = f"contrib_results_{index}"
    if contrib_key in st.session_state:
        html_content = generate_contribution_html(st.session_state[contrib_key])
        st.markdown(html_content, unsafe_allow_html=True)
    
    # Use a key that depends on the index to avoid conflicts, but we need to be careful with state
    # If we use key, streamlit manages the value.
    new_text = st.text_area("当前内容", value=display_item["文本"], height=150, key=f"edit_area_{index}")
    
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("提交", type="primary", key=f"submit_{index}", use_container_width=True):
        st.session_state.results[index]["文本"] = new_text
        if active_detector:
            # 检查是否有临时结果且文本一致，若是则直接使用
            if temp_result_key in st.session_state and st.session_state[temp_result_key]["文本"] == new_text:
                recalc = st.session_state[temp_result_key]
                st.session_state.results[index].update({
                    "AI率": recalc["AI率"],
                    "人类率": recalc["人类率"],
                    "置信度": recalc["置信度"],
                    "预测": recalc["预测"]
                })
            else:
                recalc = active_detector.detect_single(new_text)
                st.session_state.results[index].update({
                    "AI率": recalc["ai_prob"],
                    "人类率": recalc["human_prob"],
                    "置信度": recalc["confidence"],
                    "预测": recalc["prediction"]
                })
        
        # 清理状态
        st.session_state.pop(temp_result_key, None)
        st.session_state.dialog_open = False
        st.session_state.editing_index = None
        if contrib_key in st.session_state:
            del st.session_state[contrib_key]
        st.rerun()
        
    if col2.button("重算 AI率", key=f"recalc_{index}", use_container_width=True):
        if active_detector is None:
            st.error("检测器未加载")
        else:
            with st.spinner("正在重算..."):
                try:
                    recalc = active_detector.detect_single(new_text)
                    # 仅更新临时结果，不修改 item
                    new_temp = item.copy()
                    new_temp.update({
                        "文本": new_text,
                        "AI率": recalc["ai_prob"],
                        "人类率": recalc["human_prob"],
                        "置信度": recalc["confidence"],
                        "预测": recalc["prediction"]
                    })
                    st.session_state[temp_result_key] = new_temp
                    # 强制刷新以更新 metric_slot 和 text_area 的 source
                    st.rerun()
                except Exception as exc:
                    st.error(f"重算失败: {exc}")

    if col3.button("计算分布", key=f"calc_dist_{index}", use_container_width=True):
        if active_detector is None:
            st.error("检测器未加载")
        elif not new_text.strip():
            st.error("内容为空")
        else:
            lang = st.session_state.get("language_code", "chinese")
            mode = "word"
            segments = segment_text(new_text, mode=mode, language=lang)
            sentences = split_into_sentences(new_text, language=lang)
            
            dist_results = analyze_contribution_systematic(active_detector, new_text, segments, sentences, language=lang)
            st.session_state[contrib_key] = dist_results
            st.rerun()

    if col4.button("重置", key=f"reset_{index}", use_container_width=True):
        if "原文" in item:
            original = item["原文"]
            st.session_state.results[index]["文本"] = original
            st.session_state[reset_flag_key] = True
            if active_detector:
                recalc = active_detector.detect_single(original)
                st.session_state.results[index].update({
                    "AI率": recalc["ai_prob"],
                    "人类率": recalc["human_prob"],
                    "置信度": recalc["confidence"],
                    "预测": recalc["prediction"]
                })
            st.session_state.dialog_open = False
            st.session_state.editing_index = None
            if contrib_key in st.session_state:
                del st.session_state[contrib_key]
            st.rerun()


if dialog_decorator:
    @dialog_decorator("编辑内容")
    def show_edit_dialog(index):
        edit_form_content(index, st.session_state.get("detector"))


@st.cache_resource
def load_detector(language="chinese"):
    """加载检测器（缓存）"""
    with st.spinner("正在加载模型..."):
        detector = ChineseAIGCDetector(device="cpu", language=language)
    return detector


def split_into_sentences(text: str, language: str = "chinese") -> List[Tuple[str, int, int]]:
    """将文本分割成句子"""
    sentences = []
    if language == "chinese":
        pattern = r'[^。！？；!?;]+[。！？；!?;]*'
        for match in re.finditer(pattern, text):
            sentence = match.group().strip()
            if sentence:
                sentences.append((sentence, match.start(), match.end()))
    else:
        pattern = r'[^.!?]+[.!?]*'
        for match in re.finditer(pattern, text):
            sentence = match.group().strip()
            if sentence:
                sentences.append((sentence, match.start(), match.end()))
    if not sentences:
        sentences = [(text.strip(), 0, len(text))]
    return sentences

def segment_text(text: str, mode: str, language: str = "chinese") -> List[Tuple[str, int, int]]:
    """分词/分字"""
    segments = []
    if mode == "char":
        for i, char in enumerate(text):
            if char.strip():
                segments.append((char, i, i+1))
    else:
        if language == "chinese":
            words = jieba.tokenize(text)
            for word, start, end in words:
                if word.strip():
                    segments.append((word, start, end))
        else:
            pattern = r'\b\w+\b|[^\w\s]'
            for match in re.finditer(pattern, text):
                word = match.group()
                if word.strip():
                    segments.append((word, match.start(), match.end()))
    return segments

def analyze_contribution_systematic(detector: ChineseAIGCDetector, text: str, 
                                   segments: List[Tuple[str, int, int]],
                                   sentences: List[Tuple[str, int, int]],
                                   language: str = "chinese") -> List[Dict]:
    """系统性滑动窗口分析"""
    original_result = detector.detect_single(text)
    original_ai_prob = original_result["ai_prob"]
    stats = defaultdict(lambda: {"present": [], "absent": []})
    
    # Progress UI placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    window_ratios = [1/4, 1/24]
    total_iterations = 0
    for sentence_text, sent_start, sent_end in sentences:
        sent_segments = [seg for seg in segments if seg[1] >= sent_start and seg[2] <= sent_end]
        if sent_segments:
            sent_len = len(sent_segments)
            for ratio in window_ratios:
                window_size = max(1, int(sent_len * ratio))
                num_positions = sent_len + window_size - 1
                total_iterations += num_positions
    
    current_iteration = 0
    
    for sent_idx, (sentence_text, sent_start, sent_end) in enumerate(sentences):
        sent_segment_info = []
        for global_idx, seg in enumerate(segments):
            if seg[1] >= sent_start and seg[2] <= sent_end:
                sent_segment_info.append((global_idx, seg))
        
        if not sent_segment_info:
            continue
        
        sent_segments_count = len(sent_segment_info)
        sent_result = detector.detect_single(sentence_text)
        sent_original_ai_prob = sent_result["ai_prob"]
    
        for ratio in window_ratios:
            window_size = max(1, int(sent_segments_count * ratio))
            for start_pos in range(-(window_size - 1), sent_segments_count):
                window_start = max(0, start_pos)
                window_end = min(sent_segments_count, start_pos + window_size)
                if window_start >= window_end:
                    continue
                
                deleted_local_indices = set(range(window_start, window_end))
                sent_text_parts = []
                for local_idx, (global_idx, seg) in enumerate(sent_segment_info):
                    if local_idx not in deleted_local_indices:
                        sent_text_parts.append(seg[0])
                
                modified_sent_text = ''.join(sent_text_parts)
                if modified_sent_text.strip():
                    try:
                        modified_result = detector.detect_single(modified_sent_text)
                        modified_ai_prob = modified_result["ai_prob"]
                    except:
                        modified_ai_prob = sent_original_ai_prob
                else:
                    modified_ai_prob = 0
                
                for local_idx, (global_idx, seg) in enumerate(sent_segment_info):
                    if local_idx in deleted_local_indices:
                        stats[global_idx]["absent"].append(modified_ai_prob)
                    else:
                        stats[global_idx]["present"].append(modified_ai_prob)
                
                current_iteration += 1
                if total_iterations > 0:
                    progress_bar.progress(current_iteration / total_iterations)

    progress_bar.empty()
    status_text.empty()
    
    results = []
    for idx, (segment, start, end) in enumerate(segments):
        present_probs = stats[idx]["present"]
        absent_probs = stats[idx]["absent"]
        
        if len(present_probs) > 5:
            present_sorted = sorted(present_probs)
            avg_present = np.mean(present_sorted[1:-1]) if len(present_sorted) > 2 else np.mean(present_probs)
        else:
            avg_present = np.mean(present_probs) if present_probs else original_ai_prob
            
        if len(absent_probs) > 5:
            absent_sorted = sorted(absent_probs)
            avg_absent = np.mean(absent_sorted[1:-1]) if len(absent_sorted) > 2 else np.mean(absent_probs)
        else:
            avg_absent = np.mean(absent_probs) if absent_probs else original_ai_prob
        
        contribution = avg_present - avg_absent
        results.append({
            "文本": segment,
            "起始位置": start,
            "贡献度": contribution,
            "存在时AI": avg_present,
            "缺失时AI": avg_absent
        })
    return results

def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def _rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def get_contribution_color(contribution: float) -> str:
    normalized = (contribution + 0.1) / 0.2
    normalized = max(0.0, min(1.0, normalized))
    c0 = "#4caf50" 
    c1 = "#ff921e" 
    c2 = "#f23535" 
    if normalized <= 0.20: return c0
    elif normalized <= 0.50:
        t = (normalized - 0.20) / 0.30
        r0, g0, b0 = _hex_to_rgb(c0)
        r1, g1, b1 = _hex_to_rgb(c1)
        return _rgb_to_hex((_lerp(r0, r1, t), _lerp(g0, g1, t), _lerp(b0, b1, t)))
    elif normalized <= 0.60: return c1
    elif normalized <= 0.90:
        t = (normalized - 0.60) / 0.30
        r1, g1, b1 = _hex_to_rgb(c1)
        r2, g2, b2 = _hex_to_rgb(c2)
        return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))
    else: return c2

def generate_contribution_html(results: List[Dict]) -> str:
    sorted_results = sorted(results, key=lambda x: x["起始位置"])

    # 计算当前段落的最大正向贡献值，用于相对缩放
    positive_contribs = [r["贡献度"] for r in results if r.get("贡献度", 0) > 0]
    max_contrib = max(positive_contribs) if positive_contribs else 0.001

    # 连续渐变：橙 -> 红（按相对贡献 ratio 线性插值）
    c_low = "#ff921e"   # low highlight (orange)
    c_high = "#f23535"  # high highlight (red)

    def _ratio_to_color(ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        r0, g0, b0 = _hex_to_rgb(c_low)
        r1, g1, b1 = _hex_to_rgb(c_high)
        return _rgb_to_hex((_lerp(r0, r1, ratio), _lerp(g0, g1, ratio), _lerp(b0, b1, ratio)))

    html_parts = []
    for result in sorted_results:
        segment = html.escape(result["文本"])
        contribution = float(result["贡献度"])

        color = None

        # 仅处理正向贡献（忽略负向）
        if contribution > 0:
            ratio = contribution / max_contrib if max_contrib > 0 else 0.0

            # 绝对阈值 + 相对阈值：过滤噪音，但颜色在阈值以上连续渐变
            if contribution > 0.01 and ratio > 0.10:
                # 将 (0.10 ~ 1.0) 映射到 (0 ~ 1) 做连续渐变
                t = (ratio - 0.10) / 0.90
                color = _ratio_to_color(t)

        tooltip = f"字/词: {segment} | 贡献: {contribution*100:.2f}%"

        if color:
            html_parts.append(
                f'<span class="char-segment" style="border-bottom: 3px solid {color}; background-color: {color}25;" title="{tooltip}">{segment}</span>'
            )
        else:
            html_parts.append(
                f'<span class="char-segment" title="{tooltip}">{segment}</span>'
            )

    return ''.join(html_parts)


def extract_text_from_pdf(file) -> str:
    """
    从 PDF 文件提取文本
    
    Args:
        file: 上传的 PDF 文件对象
        
    Returns:
        提取的文本内容
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF 解析失败: {str(e)}")
        return ""


def extract_text_from_docx(file) -> str:
    """
    从 Word 文档提取文本
    
    Args:
        file: 上传的 Word 文件对象
        
    Returns:
        提取的文本内容
    """
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Word 文档解析失败: {str(e)}")
        return ""


def extract_chinese_from_tex(file) -> str:
    """
    从 LaTeX (.tex) 文件中提取正文内容
    只保留 document 环境内的中文/英文/数字/标点与段落
    """
    try:
        content = file.read().decode("utf-8", errors="ignore")

        # 0) 仅取 \begin{document} ... \end{document} 内的正文
        doc_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", content, flags=re.S)
        if doc_match:
            content = doc_match.group(1)

        # 1) 去注释
        content = re.sub(r"(?m)^%.*$", " ", content)

        # 2) 去\cite引用
        content = re.sub(r"\\cite\{[^}]*\}", " ", content)

        # 处理 itemize / enumerate：把每个 \item 的内容连接成单独一段（段内不保留换行）
        def _join_items(match):
            body = match.group(2)
            # 按 \item 分割并清理每项
            parts = re.split(r"\\item", body)
            items = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # 去掉内部多余空白和换行，保留内容连续性
                p = re.sub(r"\s+", " ", p)
                items.append(p)
            # 用空格连接所有 item，形成一段
            return " ".join(items)

        content = re.sub(r"\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}", _join_items, content, flags=re.S)


        # 3) 去数学公式 (行间/行内)
        math_patterns = [r"\$\$.*?\$\$", r"\\\[.*?\\\]", r"\\\(.*?\\\)", r"\$.*?\$"]
        for pat in math_patterns:
            content = re.sub(pat, " ", content, flags=re.S)

        # 4) 去环境块
        content = re.sub(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", " ", content, flags=re.S)

        # 5) 去命令名但保留括号内容
        content = re.sub(r"\\[a-zA-Z@]+(\s*\[[^\]]*\])?", " ", content)

        # 6) 去掉大括号本身，保留内容
        content = content.replace("{", "").replace("}", "")

        # 7) 保留中文、英文、数字、常见标点与换行/空白
        allowed = r"[^\u4e00-\u9fffA-Za-z0-9。，！？、；：：（）《》【】“”""''…—\-\n\r\t ,.;:!\?\(\)\[\]\{\}/'\"`]"
        content = re.sub(allowed, " ", content)

        # 8) 规范空白，保留段落
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()
    except Exception as e:
        st.error(f"TeX 解析失败: {str(e)}")
        return ""


def extract_text_from_md(file) -> str:
    """
    从 Markdown (.md) 文件中提取文本内容
    去除 Markdown 标记，保留纯文本
    """
    try:
        content = file.read().decode("utf-8", errors="ignore")
        
        # 去除代码块
        content = re.sub(r"```.*?```", " ", content, flags=re.S)
        content = re.sub(r"`[^`]+`", " ", content)
        
        # 去除图片链接
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", content)
        
        # 去除普通链接，保留文本
        content = re.sub(r"\[([^\]]*)\]\([^\)]+\)", r"\1", content)
        
        # 去除标题标记
        content = re.sub(r"^#+\s+", "", content, flags=re.M)
        
        # 去除列表标记
        content = re.sub(r"^[\*\-\+]\s+", "", content, flags=re.M)
        content = re.sub(r"^\d+\.\s+", "", content, flags=re.M)
        
        # 去除引用标记
        content = re.sub(r"^>\s+", "", content, flags=re.M)
        
        # 去除水平线
        content = re.sub(r"^[\*\-_]{3,}$", "", content, flags=re.M)
        
        # 去除粗体和斜体标记
        content = re.sub(r"\*\*([^\*]+)\*\*", r"\1", content)
        content = re.sub(r"__([^_]+)__", r"\1", content)
        content = re.sub(r"\*([^\*]+)\*", r"\1", content)
        content = re.sub(r"_([^_]+)_", r"\1", content)
        
        # 规范空白
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip()
    except Exception as e:
        st.error(f"Markdown 解析失败: {str(e)}")
        return ""


def split_into_paragraphs(text: str) -> List[str]:
    """
    将文本分割为段落
    
    Args:
        text: 输入文本
        
    Returns:
        段落列表
    """
    # 按空行
    paragraphs = re.split(r'[\n\n]', text.strip())
    # 过滤空段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def get_ai_rate_color(ai_prob: float) -> str:
    """根据 AI 概率返回对应的 CSS 类名"""
    if ai_prob > 0.75:
        return "ai-high"
    elif ai_prob > 0.5:
        return "ai-medium"
    else:
        return "ai-low"


def get_badge_class(ai_prob: float) -> str:
    """获取徽章 CSS 类名"""
    if ai_prob > 0.75:
        return "badge-high"
    elif ai_prob > 0.5:
        return "badge-medium"
    else:
        return "badge-low"


def render_ai_metric(slot, ai_prob: float):
    """在编辑对话框中以红黄绿展示当前 AI 率"""
    percent = f"{ai_prob*100:.1f}%"
    if ai_prob > 0.75:
        color = "#f23535"
        icon = "🔴 高度疑似 AI"
    elif ai_prob > 0.5:
        color = "#ff921e"
        icon = "🟡 可能 AI"
    else:
        color = "#4caf50"
        icon = "🟢 可能人类"
    slot.markdown(
        f"<div style='font-weight:600;font-size:18px;color:{color};'>当前 AI 率：{icon} | {percent}</div>",
        unsafe_allow_html=True,
    )


def format_ai_rate(ai_prob: float, human_prob: float) -> str:
    """格式化 AI 率显示"""
    ai_percent = f"{ai_prob*100:.1f}%"
    human_percent = f"{human_prob*100:.1f}%"
    
    # 确定标签
    if ai_prob > 0.75:
        label = "🔴 高度疑似 AI"
    elif ai_prob > 0.5:
        label = "🟡 可能 AI"
    else:
        label = "🟢 可能人类"
    
    return f"{label} | AI: {ai_percent} | 人类: {human_percent}"


def display_paragraph_result(para_num: int, paragraph: str, result: Dict, show_details: bool = False):
    """显示单个段落的检测结果 - 紧凑样式"""
    ai_prob = result["ai_prob"]
    human_prob = result["human_prob"]
    
    # 确定高亮样式
    if ai_prob > 0.75:
        highlight_class = "highlight-high"
        badge_class = "inline-badge-high"
        icon = "🔴"
    elif ai_prob > 0.5:
        highlight_class = "highlight-medium"
        badge_class = "inline-badge-medium"
        icon = "🟡"
    else:
        highlight_class = "highlight-low"
        badge_class = "inline-badge-low"
        icon = "🟢"
    
    # 创建紧凑的内联显示
    html_content = f"""
    <span class="text-segment {highlight_class}" title="AI率: {ai_prob*100:.1f}% | 置信度: {max(ai_prob, human_prob):.4f}">{paragraph}</span><span class="inline-badge {badge_class}">{icon}{ai_prob*100:.0f}%</span> 
    """
    
    st.markdown(html_content, unsafe_allow_html=True)


def main():
    """主函数"""
    
    # 顶部标题
    st.title("AIGC 文本检测器")
    st.markdown("### 逐段落检测 AI 生成文本")
    
    # 语言选择器
    language = st.radio(
        "选择检测语言 / Select Language",
        ("🇨🇳 中文 (Chinese)", "🇺🇸 英文 (English)"),
        horizontal=True,
        help="中文模型：yuchuantian/AIGC_detector_zhv3 | 英文模型：yuchuantian/AIGC_detector_env3"
    )
    lang_code = "chinese" if "中文" in language else "english"
    
    # 加载检测器
    detector = load_detector(language=lang_code)
    st.session_state.detector = detector
    st.session_state.language_code = lang_code

        # 预留查询参数处理（当前未使用）
    
    
    
    
    
    
    
    # 输入区域
    st.subheader("📝 输入文本")
    
    input_mode = st.radio(
        "选择输入方式",
        ("📄 直接输入文本", "📎 上传文件"),
        horizontal=True
    )
    
    text = ""
    
    if input_mode == "📄 直接输入文本":
        placeholder_text = "在这里粘贴或输入您要检测的文本..." if lang_code == "chinese" else "Paste or type the text you want to detect here..."
        text = st.text_area(
            "请输入要检测的文本 (每个段落会单独检测):",
            height=200,
            placeholder=placeholder_text,
            label_visibility="collapsed"
        )
    else:
        uploaded_file = st.file_uploader(
            "上传文件 (支持 txt, csv, pdf, docx, tex, md)",
            type=["txt", "csv", "pdf", "docx", "tex", "md"]
        )
        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            with st.spinner(f"正在解析 {file_type.upper()} 文件..."):
                if file_type == "pdf":
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        st.success(f"✓ PDF 解析完成，提取 {len(text)} 个字符")
                elif file_type in ["docx", "doc"]:
                    text = extract_text_from_docx(uploaded_file)
                    if text:
                        st.success(f"✓ Word 文档解析完成，提取 {len(text)} 个字符")
                elif file_type == "tex":
                    text = extract_chinese_from_tex(uploaded_file)
                    if text:
                        st.success(f"✓ TeX 内容提取完成，提取 {len(text)} 个字符")
                elif file_type == "md":
                    text = extract_text_from_md(uploaded_file)
                    if text:
                        st.success(f"✓ Markdown 解析完成，提取 {len(text)} 个字符")
                elif file_type == "csv":
                    text = uploaded_file.read().decode("utf-8")
                else:  # txt
                    text = uploaded_file.read().decode("utf-8")
    
    # 检测按钮
    col1, col2, col3 = st.columns(3)
    
    if col1.button("🔍 开始检测", use_container_width=False, type="primary"):
        
        if not text.strip():
            st.error("❌ 请输入文本内容")
        else:
            # 分割段落
            paragraphs = split_into_paragraphs(text)
            
            if not paragraphs:
                st.error("❌ 无法解析文本")
            else:
                st.success(f"✓ 发现 {len(paragraphs)} 个段落，正在检测...")
    
                # 检测所有段落
                progress_bar = st.progress(0)
                new_results = []
                
                for i, para in enumerate(paragraphs):
                    result = detector.detect_single(para)
                    new_results.append({
                        "段落": i + 1,
                        "文本": para,
                        "原文": para,
                        "AI率": result["ai_prob"],
                        "人类率": result["human_prob"],
                        "置信度": result["confidence"],
                        "预测": result["prediction"]
                    })
                    progress_bar.progress((i + 1) / len(paragraphs))
                
                st.session_state.results = new_results
                st.session_state.editing_index = None
                st.session_state.dialog_open = False

    # 显示结果 (如果存在)
    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # 显示图表
        st.markdown("---")
        st.subheader("统计")
        
        # 计算统计数据
        total_paragraphs = len(results)
        total_chars = sum(len(r["文本"]) for r in results)
        if total_chars == 0:
            total_chars = 1
        high_ai_count = sum(1 for r in results if r["AI率"] > 0.75)
        medium_and_high_count = sum(1 for r in results if r["AI率"] > 0.5)
        avg_ai_rate = sum(r["AI率"] * len(r["文本"]) for r in results) / total_chars
        avg_confidence = sum(r["置信度"] for r in results) / total_paragraphs
        
        # 显示统计信息
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("总段落数", f"{total_paragraphs}")
        with col2:
            st.metric("疑似及以上", f"{medium_and_high_count}")
        with col3:
            st.metric("高度疑似", f"{high_ai_count}")
        with col4:
            st.metric("平均 AI 率", f"{avg_ai_rate*100:.1f}%")
        with col5:
            st.metric("平均置信度", f"{avg_confidence:.3f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 创建一维条形图 - 所有段落在一行，宽度表示字数占比
        fig = go.Figure()
        
        # 为每个段落创建一个堆叠条
        for i, r in enumerate(results):
            # 颜色根据 AI 率（渐变：绿 -> 橙 -> 红）
            ai_rate = float(r["AI率"])

            def _lerp(a: int, b: int, t: float) -> int:
                return int(round(a + (b - a) * t))

            def _hex_to_rgb(hex_color: str):
                hex_color = hex_color.lstrip("#")
                return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

            def _rgb_to_hex(rgb):
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            def _ai_rate_to_color(p: float) -> str:
                p = max(0.0, min(1.0, p))
                c0 = "#4caf50"  # 低：绿
                c1 = "#ff921e"  # 中：橙
                c2 = "#f23535"  # 高：红
                # 分段规则（按百分比）：
                # 0~20 纯绿；20~50 变橙；50~60 纯橙；60~90 变红；90~100 纯红
                # 10~20 未指定，默认保持纯绿（与 0~10 一致）
                if p <= 0.20:
                    return c0
                if p <= 0.50:
                    t = (p - 0.20) / 0.30
                    r0, g0, b0 = _hex_to_rgb(c0)
                    r1, g1, b1 = _hex_to_rgb(c1)
                    return _rgb_to_hex((_lerp(r0, r1, t), _lerp(g0, g1, t), _lerp(b0, b1, t)))
                if p <= 0.60:
                    return c1
                if p <= 0.90:
                    t = (p - 0.60) / 0.30
                    r1, g1, b1 = _hex_to_rgb(c1)
                    r2, g2, b2 = _hex_to_rgb(c2)
                    return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))
                return c2

            color = _ai_rate_to_color(ai_rate)
            
            # 宽度根据字数占比
            width = len(r["文本"]) / total_chars * 100
            
            fig.add_trace(go.Bar(
                name=f"段落 {r['段落']}",
                x=[width],
                y=["全文"],
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(width=0)
                ),
                hovertemplate=f"<b>段落 {r['段落']}</b><br>字数: {len(r['文本'])}<br>AI率: {r['AI率']*100:.1f}%<extra></extra>"
            ))
        
        fig.update_layout(
            title="概览",
            xaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            ),
            barmode='stack',
            height=150,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    
        # 段落视图
        st.markdown("---")
        st.subheader("📄 段落视图")

        edit_mode_enabled = st.session_state.get("edit_mode_enabled", False)
        toggle_col = st.columns(1)
        with toggle_col[0]:
            if not edit_mode_enabled:
                if st.button("启用编辑", use_container_width=True):
                    st.session_state.edit_mode_enabled = True
                    st.rerun()
            else:
                if st.button("关闭编辑", use_container_width=True):
                    st.session_state.edit_mode_enabled = False
                    st.session_state.editing_index = None
                    st.session_state.dialog_open = False
                    st.rerun()
 
        # 如果对话框需要保持打开，根据状态重新展示
        if edit_mode_enabled and dialog_decorator and st.session_state.get("dialog_open") and st.session_state.get("editing_index") is not None:
            show_edit_dialog(st.session_state.editing_index)
        
        # Fallback edit area if no dialog support
        if edit_mode_enabled and not dialog_decorator and "editing_index" in st.session_state:
            idx = st.session_state.editing_index
            if idx is not None and 0 <= idx < len(results):
                with st.container():
                    st.info(f"正在编辑段落 {idx+1}")
                    edit_form_content(idx, st.session_state.get("detector"))
                    if st.button("关闭编辑", key="close_edit"):
                        del st.session_state.editing_index
                        st.rerun()

        # 显示所有段落，尾部小笔按钮触发编辑
        for i, result in enumerate(results):
            with st.container():
                col_text, col_btn = st.columns([0.985, 0.05])
                with col_text:
                    display_paragraph_result(result["段落"], result["文本"], {
                        "ai_prob": result["AI率"],
                        "human_prob": result["人类率"],
                        "confidence": result["置信度"]
                    })
                with col_btn:
                    if edit_mode_enabled and st.button("✎", key=f"btn_edit_{i} edit-btn-small", help="编辑此段落", use_container_width=True):
                        st.session_state.editing_index = i
                        if dialog_decorator:
                            st.session_state.dialog_open = True
                        st.rerun()
        
        # 显示导出按钮
        st.markdown("---")
                
        # 导出为 CSV
        csv_data = pd.DataFrame(results)
        csv_data["AI率"] = csv_data["AI率"].apply(lambda x: f"{x*100:.1f}%")
        csv_data["人类率"] = csv_data["人类率"].apply(lambda x: f"{x*100:.1f}%")
        
        csv = csv_data.to_csv(index=False)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 下载检测结果 (CSV)",
                data=csv,
                file_name="detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col2:
            # 下载已编辑后的整篇文本
            edited_full_text = "\n\n".join(r["文本"] for r in results)
            st.download_button(
                label="📄 下载已编辑文本",
                data=edited_full_text,
                file_name="edited_text.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    
    # 页脚信息
    st.markdown("---")
    model_display = "yuchuantian/AIGC_detector_zhv3" if lang_code == "chinese" else "yuchuantian/AIGC_detector_env3"
    st.markdown(f"""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>
        AIGC 检测器 v3.0 | 
        当前模型: {model_display}
        </p>
        <p>
        ⚠️ 仅供学术研究使用
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


