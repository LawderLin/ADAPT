"""
完整的 MCP CFA Agent 系統架構文檔
"""

ARCHITECTURE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    LM-AIG 系統完整架構（含 CFA Agent）                      ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                          🎯 系統核心組件                                     │
└─────────────────────────────────────────────────────────────────────────────┘

1. ItemWritingAgent (題目生成)
   ├─ 輸入: 測驗規格 (str)
   ├─ 處理: 使用 LLM 生成心理測驗題目
   ├─ 輸出: JSON 格式的題目
   └─ 技術: Ollama + llama3.2:1b

2. CriticAgent (多層次評審)
   ├─ ContentReviewer: 內容效度評估
   ├─ LinguisticReviewer: 語言品質檢查
   ├─ BiasReviewer: 偏見檢測
   ├─ MetaReviewer: 結果整合
   └─ 輸出: 評分 (1-10) + 改進建議

3. DataAnalysisPipeline (探索性因素分析)
   ├─ 資料載入與預處理
   ├─ EFA (Exploratory Factor Analysis)
   ├─ 信度分析 (Cronbach's Alpha)
   ├─ 項目統計
   └─ 輸出: EFA 報告

4. ⭐ CFAAgent (驗證性因素分析) [新增]
   ├─ 資料預處理
   ├─ CFA (Confirmatory Factor Analysis)
   ├─ 因素數量判斷 (Kaiser + 方差準則)
   ├─ LLM 因素命名
   ├─ 品質評估
   ├─ 題目改進建議
   └─ 輸出: 完整 CFA 報告

5. LM_AIG_System (整體工作流程)
   ├─ 迭代式題目生成
   ├─ 質量評審
   ├─ 資料分析
   └─ 報告生成

┌─────────────────────────────────────────────────────────────────────────────┐
│                     📊 完整的測驗開發工作流程                                   │
└─────────────────────────────────────────────────────────────────────────────┘

步驟 1: 需求分析
   └─→ 定義測驗的理論背景和核心構念

步驟 2: 題目生成 (ItemWritingAgent)
   ├─ 輸入: 測驗規格
   ├─ 處理: AI 自動生成題目
   └─ 輸出: 初始題目集

步驟 3: 質量評審 (CriticAgent)
   ├─ 內容評審
   ├─ 語言評審
   ├─ 偏見檢查
   └─ 迭代改進 (評分 < 7 時)

步驟 4: 資料收集
   └─→ 招募受試者並收集反應資料

步驟 5: 探索性分析 (DataAnalysisPipeline)
   ├─ EFA 確定初步因素結構
   ├─ 信度分析
   └─ 題目篩選

步驟 6: 驗證性分析 ⭐ (CFAAgent)
   ├─ CFA 驗證因素結構
   ├─ LLM 基於理論背景命名因素
   ├─ 品質評估
   ├─ 題目改進建議 (刪除/修改)
   └─ 輸出: 最終測驗版本

步驟 7: 報告生成
   └─→ 自動生成完整的技術報告

┌─────────────────────────────────────────────────────────────────────────────┐
│                    🔄 CFAAgent 的內部流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

analyze() 方法流程：

輸入: 理論背景 + 資料 + [因素數量]
  │
  ├─→ 1️⃣ 資料預處理
  │   └─ 標準化 + 缺值處理
  │
  ├─→ 2️⃣ 因素數量判斷 (若未指定)
  │   ├─ Kaiser 準則 (特徵值 > 1)
  │   ├─ 方差準則 (解釋 70% 方差)
  │   └─ 取平均值作為推薦
  │
  ├─→ 3️⃣ 執行 CFA
  │   ├─ FactorAnalysis 計算
  │   ├─ 因素負荷量矩陣
  │   ├─ 解釋方差比例
  │   └─ Cronbach's Alpha
  │
  ├─→ 4️⃣ LLM 因素命名
  │   ├─ 若有 Ollama: 使用 LLM 智能命名
  │   └─ 若無: 使用預設命名 (Factor_1, Factor_2...)
  │
  ├─→ 5️⃣ 品質評估
  │   ├─ 信度評級 (α 值)
  │   ├─ 負荷量品質
  │   └─ 整體評級 (Excellent/Good/Fair/Poor)
  │
  ├─→ 6️⃣ 題目改進建議
  │   ├─ 高品質 (|loading| ≥ 0.5): 保留
  │   ├─ 中等品質 (0.3-0.5): 審視/修改
  │   └─ 低品質 (< 0.3): 考慮刪除
  │
  └─→ 7️⃣ 生成報告
      └─ 詳細的文字分析報告

輸出: 完整結果字典
  ├─ status: "success"
  ├─ n_factors: 因素數量
  ├─ factor_names: 因素命名
  ├─ cfa_results: 統計結果
  ├─ quality_assessment: 品質評估
  ├─ item_suggestions: 題目改進建議
  └─ report: 文字報告

┌─────────────────────────────────────────────────────────────────────────────┐
│                      📁 檔案結構和關鍵文件                                    │
└─────────────────────────────────────────────────────────────────────────────┘

251204_LTCL_final/
├── mcp_cfa_agent.py
│   └─ CFAAgent 主類 (驗證性因素分析)
│      ├─ analyze() - 主要分析方法
│      ├─ _perform_cfa() - 執行 CFA
│      ├─ _name_factors() - LLM 因素命名
│      ├─ _assess_factor_quality() - 品質評估
│      ├─ _generate_item_suggestions() - 題目建議
│      └─ _generate_comprehensive_report() - 報告生成
│
├── data_analysis/
│   └── pipeline.py
│       └─ DataAnalysisPipeline (探索性分析)
│          ├─ load_simulated_data()
│          ├─ exploratory_factor_analysis()
│          ├─ _calculate_cronbach_alpha()
│          └─ _calculate_item_statistics()
│
├── test_cfa_agent.py
│   └─ CFA Agent 整合測試腳本
│
├── example_cfa_notebook.py
│   └─ Jupyter notebook 使用示例
│
├── CFA_AGENT_GUIDE.md
│   └─ 完整使用文檔
│
└── ARCHITECTURE.md
    └─ 本檔案

┌─────────────────────────────────────────────────────────────────────────────┐
│                     💡 使用示例速查                                          │
└─────────────────────────────────────────────────────────────────────────────┘

# 快速開始
from mcp_cfa_agent import CFAAgent
import pandas as pd

data = pd.read_csv('survey_responses.csv')
agent = CFAAgent(ollama_client=None)

results = agent.analyze(
    theoretical_background="測驗的理論背景...",
    data=data,
    n_factors=3
)

# 查看結果
print(results['report'])
print(results['factor_names'])
print(results['item_suggestions'])

# 進階：使用 LLM 進行因素命名
import ollama
agent = CFAAgent(ollama_client=ollama)
results = agent.analyze(
    theoretical_background="詳細的理論背景說明...",
    data=data
)

┌─────────────────────────────────────────────────────────────────────────────┐
│                     📊 輸出數據格式詳解                                       │
└─────────────────────────────────────────────────────────────────────────────┘

results 字典結構：

{
  "status": "success",
  
  "n_factors": 3,
  
  "factor_names": {
    "Factor_1": "學習自我效能",
    "Factor_2": "社交自我效能",
    "Factor_3": "挑戰應對自我效能"
  },
  
  "cfa_results": {
    "cronbach_alpha": 0.87,  # 0-1，越高越好
    "loadings": {             # 因素負荷量
      "Item_1": {"Factor_1": 0.72, "Factor_2": 0.15, ...},
      ...
    },
    "variance_explained": [0.35, 0.33, 0.32],  # 各因素解釋的方差比
    "correlation_matrix": {...}
  },
  
  "quality_assessment": {
    "overall_quality": "Excellent",  # Excellent/Good/Fair/Poor
    "reliability": {
      "cronbach_alpha": 0.87,
      "interpretation": "良好 (0.8 ≤ α < 0.9)"
    },
    "loading_quality": {
      "Factor_1": {
        "high_loading_items": 4,     # 高負荷量題目數
        "mean_loading": 0.68,        # 平均負荷量
        "quality": "Excellent"       # 品質評級
      },
      ...
    }
  },
  
  "item_suggestions": {
    "items_to_remove": [          # 建議刪除（|loading| < 0.3）
      "Item_5 (最高負荷量: 0.25)"
    ],
    "items_to_review": [          # 建議審視（0.3 ≤ |loading| < 0.5）
      "Item_8 (最高負荷量: 0.42)"
    ],
    "items_to_keep": [            # 保留（|loading| ≥ 0.5）
      "Item_1 (最高負荷量: 0.72)",
      ...
    ]
  },
  
  "report": "📊 驗證性因素分析 (CFA) 詳細報告\n..."  # 文字報告
}

┌─────────────────────────────────────────────────────────────────────────────┐
│                      🔗 與其他組件的集成                                      │
└─────────────────────────────────────────────────────────────────────────────┘

# 與 ItemWritingAgent 集成
from mcp_cfa_agent import CFAAgent
from 計算語言學期末——Agent_Workflow import ItemWritingAgent

# 先生成題目
writer = ItemWritingAgent()
generated = writer.generate_items("規格...", num_items=12)

# 再進行 CFA 驗證
cfa_agent = CFAAgent()
# ... 使用真實或模擬資料進行分析

# 與 DataAnalysisPipeline 集成
from data_analysis.pipeline import DataAnalysisPipeline

pipeline = DataAnalysisPipeline()
data = pipeline.load_simulated_data(n_items=12)

# 先進行 EFA
efa_results = pipeline.exploratory_factor_analysis(n_factors=3)

# 再進行 CFA
cfa_results = cfa_agent.analyze("理論背景...", data, n_factors=3)

┌─────────────────────────────────────────────────────────────────────────────┐
│                      🎯 核心指標解釋                                         │
└─────────────────────────────────────────────────────────────────────────────┘

1. Cronbach's Alpha (信度係數)
   範圍: 0 - 1
   解釋:
   - α ≥ 0.9: 優異（可能過度相關）
   - 0.8 ≤ α < 0.9: 良好 ✅
   - 0.7 ≤ α < 0.8: 可接受
   - 0.6 ≤ α < 0.7: 勉強可接受
   - α < 0.6: 不可接受 ❌

2. 因素負荷量 (Factor Loading)
   範圍: -1 到 1
   解釋:
   - |loading| ≥ 0.7: 優異
   - 0.5 ≤ |loading| < 0.7: 良好 ✅
   - 0.3 ≤ |loading| < 0.5: 中等
   - |loading| < 0.3: 低弱 ❌

3. 解釋方差比例 (Variance Explained)
   範圍: 0% - 100%
   目標: ≥ 70% （通常認為可接受）

4. 平均因素負荷量
   目標: ≥ 0.5 （表示因素解釋良好）

┌─────────────────────────────────────────────────────────────────────────────┐
│                      🚀 效能和優化                                           │
└─────────────────────────────────────────────────────────────────────────────┘

時間複雜度: O(n × p²)
- n: 受試者數
- p: 題目數

典型運行時間:
- 300 人 × 12 題: < 1 秒
- 1000 人 × 50 題: 1-5 秒
- + LLM 因素命名: 額外 5-30 秒

優化建議:
- 大規模資料 (> 5000 人): 考慮分層抽樣
- 批量分析: 保留 LLM 客戶端連接以提高效率
- 無 LLM 時: 系統自動使用預設命名，提升速度

╔════════════════════════════════════════════════════════════════════════════╗
║                           🎉 系統已完成！                                   ║
║                                                                             ║
║  包含: 題目生成、多層評審、探索性分析、驗證性分析和智能報告生成            ║
║  技術: Ollama LLM + scikit-learn + pandas                                  ║
║  狀態: ✅ 測試完成，可投入使用                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(ARCHITECTURE)
