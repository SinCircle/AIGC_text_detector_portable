"""
Streamlit 前端 - 段落级别 AIGC 检测工具
支持逐段落检测和可视化显示
"""

import streamlit as st
import pandas as pd
from advanced_detector import ChineseAIGCDetector
import plotly.graph_objects as go
from typing import List, Dict
import re
import PyPDF2
from docx import Document
import io

# 页面配置
st.set_page_config(
    page_title="AIGC 中文检测器",
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
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(language="chinese"):
    """加载检测器（缓存）"""
    with st.spinner("正在加载模型..."):
        detector = ChineseAIGCDetector(device="cpu", language=language)
    return detector


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
        allowed = r"[^\u4e00-\u9fffA-Za-z0-9。，！？、；：：（）《》【】""''…—\-\n\r\t ,.;:!\?\(\)\[\]\{\}/'\"`]"
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
    # 按空行或多个句号分割
    paragraphs = re.split(r'[\n\n]+|(?<=[。！？])\s+', text.strip())
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
            type=["txt", "csv", "pdf", "docx", "doc", "tex", "md"]
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
                        st.success(f"✓ TeX 中文内容提取完成，提取 {len(text)} 个字符")
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
                results = []
                
                for i, para in enumerate(paragraphs):
                    result = detector.detect_single(para)
                    results.append({
                        "段落": i + 1,
                        "文本": para,
                        "AI率": result["ai_prob"],
                        "人类率": result["human_prob"],
                        "置信度": result["confidence"],
                        "预测": result["prediction"]
                    })
                    progress_bar.progress((i + 1) / len(paragraphs))
            
            # 显示图表
            if len(results) > 0:
                st.markdown("---")
                st.subheader("统计")
                
                # 计算统计数据
                total_paragraphs = len(results)
                high_ai_count = sum(1 for r in results if r["AI率"] > 0.75)
                medium_and_high_count = sum(1 for r in results if r["AI率"] > 0.5)
                avg_ai_rate = sum(r["AI率"] for r in results) / total_paragraphs
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
                
                total_chars = sum(len(r["文本"]) for r in results)
                
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
                
                # 显示所有段落
                for result in results:
                    display_paragraph_result(result["段落"], result["文本"], {
                        "ai_prob": result["AI率"],
                        "human_prob": result["人类率"],
                        "confidence": result["置信度"]
                    })
                
                # 显示导出按钮
                st.markdown("---")
                
                # 导出为 CSV
                csv_data = pd.DataFrame(results)
                csv_data["AI率"] = csv_data["AI率"].apply(lambda x: f"{x*100:.1f}%")
                csv_data["人类率"] = csv_data["人类率"].apply(lambda x: f"{x*100:.1f}%")
                
                csv = csv_data.to_csv(index=False)
                
                st.download_button(
                    label="📥 下载检测结果 (CSV)",
                    data=csv,
                    file_name="detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    
    # 页脚信息
    st.markdown("---")
    model_display = "yuchuantian/AIGC_detector_zhv3" if lang_code == "chinese" else "yuchuantian/AIGC_detector_env3"
    st.markdown(f"""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>
        AIGC 检测器 v3.0 | 
        当前模型: {model_display} | 
        准确率: 97%+
        </p>
        <p>
        📌 提示: 置信度越高，预测越可靠 | 
        ⚠️ 仅供学术研究使用
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
