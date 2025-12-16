"""
高级中文 AIGC 文本检测器
支持批量检测、文件处理、详细统计分析
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple

class ChineseAIGCDetector:
    """中文 AIGC 检测器类"""
    
    def __init__(self, model_name: str = "yuchuantian/AIGC_detector_zhv3", device: str = "cpu", language: str = "chinese"):
        """
        初始化检测器
        
        Args:
            model_name: HuggingFace 模型名称（如未指定，会根据 language 自动选择）
            device: 计算设备 ('cpu' 或 'cuda')
            language: 语言类型 ('chinese' 或 'english')
        """
        # 根据语言自动选择模型（如果用户没有明确指定）
        if model_name == "yuchuantian/AIGC_detector_zhv3" and language == "english":
            model_name = "yuchuantian/AIGC_detector_env3"
        self.model_name = model_name
        self.device = device
        
        # 如果仓库根目录下存在 models/<模型名> 目录，优先从本地加载（方便制作携带版）
        local_models_root = Path(os.path.dirname(__file__)) / "models"
        local_candidate = local_models_root / Path(model_name.split('/')[-1])
        if local_candidate.exists():
            print(f"发现本地模型：{local_candidate}，将从本地加载")
            model_name = str(local_candidate)

        print(f"正在加载模型 {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        print("✓ 模型加载完成")
    
    def detect_single(self, text: str) -> Dict:
        """
        检测单个文本
        
        Args:
            text: 要检测的文本
            
        Returns:
            包含预测结果的字典
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            human_prob = probabilities[0][0].item()
            ai_prob = probabilities[0][1].item()
            prediction = 1 if ai_prob > human_prob else 0
            
            return {
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "full_text": text,
                "prediction": "AI生成" if prediction == 1 else "人类撰写",
                "human_prob": round(human_prob, 4),
                "ai_prob": round(ai_prob, 4),
                "confidence": round(max(human_prob, ai_prob), 4)
            }
    
    def detect_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        批量检测多个文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            检测结果列表
        """
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                result = self.detect_single(text)
                results.append(result)
                
                current = min(i + len(batch_texts), total)
                print(f"进度: {current}/{total}", end="\r")
        
        print(f"进度: {total}/{total} ✓")
        return results
    
    def detect_file(self, file_path: str) -> Dict:
        """
        检测文件中的文本
        
        Args:
            file_path: 文件路径 (支持 .txt, .json, .csv)
            
        Returns:
            检测结果统计
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        texts = []
        
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item if isinstance(item, str) else item.get("text", "") for item in data]
                else:
                    texts = [data.get("text", "")]
        
        elif file_path.suffix == ".csv":
            import csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                texts = [row[0] for row in reader if row]
        
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        print(f"从文件读取了 {len(texts)} 条文本")
        results = self.detect_batch(texts)
        
        return self._summarize_results(results)
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """
        统计分析检测结果
        
        Args:
            results: 检测结果列表
            
        Returns:
            统计摘要
        """
        ai_count = sum(1 for r in results if r["prediction"] == "AI生成")
        human_count = len(results) - ai_count
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        return {
            "总文本数": len(results),
            "AI生成": ai_count,
            "人类撰写": human_count,
            "AI占比": f"{100*ai_count/len(results):.2f}%",
            "平均置信度": round(avg_confidence, 4),
            "详细结果": results
        }


def main():
    """主函数 - 交互式使用示例"""
    
    # 初始化检测器
    detector = ChineseAIGCDetector()
    
    print("\n" + "="*60)
    print("中文 AIGC 文本检测器")
    print("="*60 + "\n")
    
    # 示例 1: 单个文本检测
    print("【示例 1】单个文本检测:")
    print("-" * 60)
    
    test_text = "作为一个人工智能助手，我很高兴能够帮助你。这是一个示例文本。"
    result = detector.detect_single(test_text)
    
    print(f"文本: {result['text']}")
    print(f"预测: {result['prediction']}")
    print(f"人类概率: {result['human_prob']:.4f}")
    print(f"AI概率: {result['ai_prob']:.4f}")
    print(f"置信度: {result['confidence']:.4f}\n")
    
    # 示例 2: 批量检测
    print("【示例 2】批量检测:")
    print("-" * 60)
    
    test_texts = [
        "今天天气真好，我出去散步了。",
        "在这个快速发展的时代，我们需要不断学习和进步。",
        "机器学习是人工智能的重要分支，广泛应用于各个领域。"
    ]
    
    batch_results = detector.detect_batch(test_texts)
    summary = detector._summarize_results(batch_results)
    
    print(f"检测结果统计:")
    print(f"  总文本数: {summary['总文本数']}")
    print(f"  AI生成: {summary['AI生成']}")
    print(f"  人类撰写: {summary['人类撰写']}")
    print(f"  AI占比: {summary['AI占比']}")
    print(f"  平均置信度: {summary['平均置信度']}\n")
    
    # 示例 3: 保存结果
    print("【示例 3】保存检测结果:")
    print("-" * 60)
    
    output_file = "detection_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 结果已保存到: {output_file}\n")


if __name__ == "__main__":
    main()
