"""
CFA Agent 整合測試
展示如何使用 MCP CFA Agent 進行驗證性因素分析
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))

from data_analysis.cfa_agent import CFAAgent


def generate_test_data(n_participants: int = 300, n_items: int = 12, n_factors: int = 3) -> pd.DataFrame:
    """
    生成測試資料：模擬有 3 個隱藏因素的測驗反應
    """
    np.random.seed(42)
    
    # 生成潛在因素分數
    factor_scores = np.random.normal(0, 1, (n_participants, n_factors))
    
    # 因素負荷量矩陣
    loadings = np.array([
        [0.8, 0.1, 0.1],  # Item 1-4: 因素 1
        [0.75, 0.15, 0.1],
        [0.7, 0.2, 0.15],
        [0.72, 0.18, 0.12],
        
        [0.1, 0.85, 0.05],  # Item 5-8: 因素 2
        [0.15, 0.8, 0.1],
        [0.05, 0.78, 0.15],
        [0.12, 0.82, 0.08],
        
        [0.1, 0.05, 0.8],   # Item 9-12: 因素 3
        [0.15, 0.1, 0.82],
        [0.08, 0.12, 0.79],
        [0.12, 0.08, 0.81]
    ])
    
    # 生成題目反應
    responses = factor_scores @ loadings.T + np.random.normal(0, 0.3, (n_participants, n_items))
    
    # 轉換為 5 點李克特量表
    responses = np.clip(np.round(3 + responses), 1, 5)
    
    # 建立 DataFrame
    data = pd.DataFrame(
        responses,
        columns=[f'Item_{i+1}' for i in range(n_items)]
    )
    
    return data


def test_cfa_agent_without_ollama():
    """
    測試 CFA Agent（無 LLM 進行因素命名）
    """
    print("🧪 開始測試 CFA Agent (無 LLM)...")
    print("="*70)
    
    # 生成測試資料
    print("\n📊 生成測試資料...")
    data = generate_test_data(n_participants=300, n_items=12, n_factors=3)
    print(f"資料大小: {data.shape}")
    print(f"資料樣本:\n{data.head()}")
    
    # 初始化 CFA Agent (不使用 Ollama)
    cfa_agent = CFAAgent()
    
    # 理論背景
    theoretical_background = """
    這是一個測量個體心理特性的測驗，包含三個主要因素：
    1. 認知能力：測量個體的學習能力和問題解決能力
    2. 情感調節：測量個體控制和調節自身情緒的能力
    3. 社交適應：測量個體與他人互動和適應社會環境的能力
    """
    
    # 執行分析
    print("\n" + "="*70)
    print("🔍 執行驗證性因素分析...")
    print("="*70)
    
    results = cfa_agent.analyze(
        theoretical_background=theoretical_background,
        data=data,
        n_factors=3,
        name_with_llm=False
    )
    
    # 顯示結果
    print("\n" + "="*70)
    print("📋 分析結果摘要")
    print("="*70)
    
    print(f"\n✅ 分析狀態: {results['status']}")
    print(f"🔢 因素數量: {results['n_factors']}")
    print(f"🏷️  因素名稱: {results['factor_names']}")
    
    print(f"\n📈 信度分析:")
    print(f"  Cronbach's Alpha: {results['cfa_results']['cronbach_alpha']:.4f}")
    
    print(f"\n⭐ 整體品質評級: {results['quality_assessment']['overall_quality']}")
    
    print(f"\n✏️  題目改進建議:")
    print(f"  • 建議刪除: {len(results['item_suggestions']['items_to_remove'])} 個題目")
    print(f"  • 建議審視: {len(results['item_suggestions']['items_to_review'])} 個題目")
    print(f"  • 保留: {len(results['item_suggestions']['items_to_keep'])} 個題目")
    
    # 顯示完整報告
    print("\n" + results['report'])
    
    return results


def test_cfa_agent_with_ollama():
    """
    測試 CFA Agent（使用 LLM 進行因素命名）
    """
    
    print("\n\n🧪 開始測試 CFA Agent (使用 LLM)...")
    print("="*70)
    
    # 生成測試資料
    print("\n📊 生成測試資料...")
    data = generate_test_data(n_participants=300, n_items=12, n_factors=3)
    
    # 初始化 CFA Agent (使用 Ollama)
    cfa_agent = CFAAgent()
    
    # 理論背景
    theoretical_background = """
    自我效能量表 (Self-Efficacy Scale)
    
    理論基礎：Bandura 的自我效能理論，測量個體對完成特定任務的信心程度。
    
    包含三個主要維度：
    1. 學習自我效能：面對學習挑戰時的信心
    2. 社交自我效能：在社交互動中的信心
    3. 挑戰應對自我效能：面對困難和挫折時的信心
    """

    # 模擬題目列表共12題
    test_items = [
        "我相信自己能夠克服學習上的困難。",
        "我在社交場合中感到自信。",
        "當面對挑戰時，我能保持冷靜並找到解決方法。",
        "我有能力完成我設定的學習目標。",
        "我能夠有效地與他人溝通和互動。",
        "遇到學習障礙時，我會積極尋求解決方案。",
        "我在陌生環境中能夠自如地表達自己。",
        "面對壓力時，我能夠調整自己的情緒。",
        "我相信自己能夠適應新的學習內容。",
        "我能夠在團隊合作中發揮積極作用。",
        "遇到困難時，我不會輕易放棄。",
        "我能夠主動建立和維持良好的人際關係。",
    ]

    # 執行分析
    print("\n" + "="*70)
    print("🔍 執行驗證性因素分析（含 LLM 因素命名）...")
    print("="*70)
    
    results = cfa_agent.analyze(
        theoretical_background=theoretical_background,
        test_items=test_items,
        data=data,
        n_factors=3
    )
    
    # 顯示結果
    print("\n" + "="*70)
    print("📋 LLM 因素命名結果")
    print("="*70)
    
    print(f"\n🏷️  LLM 提供的因素命名:")
    for factor, name in results['factor_names'].items():
        print(f"  • {factor}: {name}")
    
    print("\n" + results['report'])
    
    return results


if __name__ == "__main__":
    # 測試 1：無 LLM
    # print("\n" + "🎯 測試 1：基礎 CFA 分析 (無 LLM)")
    # results1 = test_cfa_agent_without_ollama()
    
    # 測試 2：使用 LLM
    print("\n\n" + "🎯 測試：整合 LLM 的 CFA 分析")
    try:
        import ollama
        results2 = test_cfa_agent_with_ollama()
    except ImportError:
        print("⚠️  ollama 套件不可用，跳過 LLM 測試")
    
    print("\n\n" + "="*70)
    print("✅ 所有測試完成！")
    print("="*70)
