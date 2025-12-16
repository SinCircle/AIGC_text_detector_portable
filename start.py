#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit 应用启动脚本
"""

import subprocess
import sys
import os

def main():
    """主函数"""
    
    # 检查依赖
    print("✓ 检查依赖...")
    
    required_packages = [
        ("torch", "torch", "PyTorch"),
        ("transformers", "transformers", "Transformers"),
        ("safetensors", "safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece", "SentencePiece"),
        ("streamlit", "streamlit", "Streamlit"),
        ("plotly", "plotly", "Plotly"),
        ("pandas", "pandas", "Pandas"),
        ("PyPDF2", "PyPDF2", "PyPDF2"),
        ("python-docx", "docx", "python-docx"),
        ("huggingface-hub", "huggingface_hub", "huggingface-hub"),
    ]

    missing_packages = []

    for pip_name, import_name, display_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {display_name} 已安装")
        except ImportError:
            print(f"  ✗ {display_name} 未安装")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_packages)}")
        print("正在安装...")
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            *missing_packages
        ])
        
        print("✓ 依赖安装完成\n")
    
    # 启动 Streamlit 应用
    print("🚀 启动 Streamlit 应用...\n")
    print("浏览器会自动打开 http://localhost:8501")
    print("按 Ctrl+C 停止应用\n")
    
    # 运行 Streamlit
    app_path = os.path.join(os.path.dirname(__file__), 'app_streamlit.py')
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            app_path,
            '--client.showErrorDetails=true'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止，再见！")
        sys.exit(0)

if __name__ == '__main__':
    main()
