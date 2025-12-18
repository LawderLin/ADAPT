"""
CFA Agent - 驗證性因素分析與因素命名代理人

功能：
1. 執行驗證性因素分析 (CFA)
2. 基於理論背景進行因素命名
3. 生成詳細的分析報告
4. 提供題目刪改建議
5. 推薦最佳因素數量與名稱
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from config import config

class CFAAgent:
    """
    執行因素分析、因素命名、以及提供測驗改進建議
    """
    
    def __init__(self):
        """
        初始化 CFA 代理人
        """
        self.model = config.model_name
        self.system_prompt = """You are an expert in psychometrics and factor analysis. Your task is to assist in performing Confirmatory Factor Analysis (CFA) on psychological test data, naming the factors based on theoretical background, and providing suggestions for test item improvement."""
        self.cfa_results = None
        self.data = None
        self.scaler = StandardScaler()
        
    def analyze(self, 
                theoretical_background: str,
                test_items: list[str],
                data: pd.DataFrame,
                n_factors: Optional[int] = None,
                name_with_llm: bool = True) -> Dict[str, Any]:
        """
        執行完整的驗證性因素分析流程
        
        Args:
            theoretical_background: 測驗的理論背景
            data: 測驗原始數據 (DataFrame)
            n_factors: 因素數量 (若為 None 則自動判斷)
            
        Returns:
            包含分析結果、建議和因素命名的完整報告
        """
        print("🔍 開始驗證性因素分析流程...")
        
        self.data = data
        
        # 1. 資料預處理
        print("📊 資料預處理中...")
        processed_data = self._preprocess_data(data)
        
        # 2. 確定因素數量
        if n_factors is None:
            print("🔢 自動判斷因素數量...")
            n_factors = self._determine_optimal_factors(processed_data)
            print(f"✅ 推薦因素數量: {n_factors}")
        
        # 3. 執行因素分析
        print("📈 執行因素分析...")
        cfa_results = self._perform_cfa(processed_data, n_factors)
        
        # 4. 因素命名 (使用 LLM)
        print("🏷️  進行因素命名...")
        factor_names = self._name_factors(
            theoretical_background=theoretical_background,
            test_items=test_items,
            loadings=cfa_results['loadings_df'],
            n_factors=n_factors,
            name_with_llm=name_with_llm
        )
        
        # 5. 評估因素品質
        print("⭐ 評估因素品質...")
        quality_assessment = self._assess_factor_quality(cfa_results)
        
        # 6. 生成題目改進建議
        print("✏️  生成題目改進建議...")
        item_suggestions = self._generate_item_suggestions(
            cfa_results,
            factor_names
        )
        
        # 7. 生成完整報告
        print("📋 生成分析報告...")
        report = self._generate_comprehensive_report(
            theoretical_background=theoretical_background,
            cfa_results=cfa_results,
            factor_names=factor_names,
            quality_assessment=quality_assessment,
            item_suggestions=item_suggestions,
            n_factors=n_factors
        )
        
        # 組合最終結果
        final_results = {
            "status": "success",
            "theoretical_background": theoretical_background,
            "n_factors": n_factors,
            "factor_names": factor_names,
            "cfa_results": {
                "cronbach_alpha": cfa_results['cronbach_alpha'],
                "loadings": cfa_results['loadings_df'].round(3).to_dict(),
                "variance_explained": cfa_results['explained_variance'],
                "correlation_matrix": cfa_results['correlation_matrix'].round(3).to_dict()
            },
            "quality_assessment": quality_assessment,
            "item_suggestions": item_suggestions,
            "report": report
        }
        
        print("✅ 分析完成！")
        
        return final_results
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        資料預處理：標準化和缺值處理
        """
        # 處理缺值
        data_clean = data.dropna()
        
        if len(data_clean) < len(data):
            print(f"⚠️  移除了 {len(data) - len(data_clean)} 列含缺值的資料")
        
        # 標準化
        scaled_data = self.scaler.fit_transform(data_clean)
        
        return scaled_data
    
    def _determine_optimal_factors(self, data: np.ndarray) -> int:
        """
        基於碎石圖 (Scree Plot) 和 Kaiser 準則判斷最佳因素數量
        """
        # 計算特徵值
        correlation_matrix = np.corrcoef(data.T)
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Kaiser 準則：特徵值 > 1
        kaiser_criterion = np.sum(eigenvalues > 1)
        
        # 方差解釋準則：至少解釋 70% 的方差
        cumsum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        variance_criterion = np.argmax(cumsum_variance >= 0.7) + 1
        
        # 取兩個準則的平均值
        optimal_factors = max(1, int(np.round((kaiser_criterion + variance_criterion) / 2)))
        
        return min(optimal_factors, data.shape[1] - 1)
    
    def _perform_cfa(self, data: np.ndarray, n_factors: int) -> Dict[str, Any]:
        """
        執行因素分析
        """
        # 因素分析
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        factor_scores = fa.fit_transform(data)
        
        # 因素負荷量
        loadings = fa.components_.T
        
        # 計算解釋方差
        explained_variance = []
        for i in range(n_factors):
            var = np.var(factor_scores[:, i]) if factor_scores.shape[1] > i else 0
            explained_variance.append(var)
        
        total_variance = np.sum(explained_variance)
        variance_ratio = [v / total_variance for v in explained_variance] if total_variance > 0 else [0] * n_factors
        
        # Cronbach's Alpha
        alpha = self._calculate_cronbach_alpha(data)
        
        # 相關矩陣
        correlation_matrix = np.corrcoef(data.T)
        
        # 建立負荷量 DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'Factor_{i+1}' for i in range(n_factors)],
            index=[f'Item_{i+1}' for i in range(data.shape[1])]
        )
        
        return {
            "loadings": loadings,
            "loadings_df": loadings_df,
            "factor_scores": factor_scores,
            "explained_variance": variance_ratio,
            "correlation_matrix": pd.DataFrame(correlation_matrix),
            "cronbach_alpha": alpha,
            "n_factors": n_factors
        }
    
    def _calculate_cronbach_alpha(self, data: np.ndarray) -> float:
        """
        計算 Cronbach's Alpha 信度係數
        """
        k = data.shape[1]
        if k < 2:
            return 0.0
        
        item_variances = np.var(data, axis=0, ddof=1)
        total_variance = np.var(np.sum(data, axis=1), ddof=1)
        
        if total_variance == 0:
            return 0.0
        
        alpha = (k / (k - 1)) * (1 - np.sum(item_variances) / total_variance)
        
        return max(0, min(1, alpha))  # 限制在 0-1 之間
    
    def _name_factors(self,
                     theoretical_background: str,
                     test_items: list[str],
                     loadings: pd.DataFrame,
                     n_factors: int,
                     name_with_llm: bool = True) -> Dict[str, str]:
        """
        使用 LLM 根據理論背景和因素負荷量命名因素
        """
        if not name_with_llm:
            return {f"Factor_{i+1}": f"Factor_{i+1}" for i in range(n_factors)}
        
        try:
            import ollama
        except ImportError:
            print("⚠️  無 Ollama 客戶端，使用預設因素命名")
            return {f"Factor_{i+1}": f"Factor_{i+1}" for i in range(n_factors)}
        
        # 提取各因素最高負荷量的項目
        factor_descriptions = []
        for idx, factor_col in enumerate(loadings.columns, 1):
            # 找出負荷量最高的 3-5 個題目
            top_k = min(5, len(loadings))
            top_indices = loadings[factor_col].abs().nlargest(top_k).index
            top_descriptions = []
            for item_idx in top_indices:
                loading = loadings.loc[item_idx, factor_col]
                item_text = test_items[int(item_idx.split('_')[1]) - 1] if len(test_items) > 0 else f"{item_idx}"
                top_descriptions.append(f"    - {item_text[:80]} (負荷量: {loading:.3f})")
            
            description = f"\nFactor_{idx} 的主要題目：\n" + "\n".join(top_descriptions)
            factor_descriptions.append(description)
        
        prompt = f"""你是心理測驗專家。根據以下理論背景和因素分析結果，請為每個因素命名。

理論背景：
{theoretical_background}

因素分析結果：
{''.join(factor_descriptions)}

請依序為每個因素提供一個簡潔的中文名稱（2-6 字），直接輸出名稱列表。
格式例如：
多方思考
整合性
和諧性

只輸出因素名稱，每個名稱占一行。
"""
        
        factor_names = {}
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是專業的心理測驗分析師，擅長根據題項內容和理論背景為潛在因素命名。"},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3, "num_predict": 500},
                think=False
            )
            
            content = response['message']['content'].strip()
            
            # 簡單解析：每行一個因素名稱
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            names = [line for line in lines if line and len(line) <= 20 and not line.startswith('#')]
            
            # 映射因素名稱
            for i in range(min(n_factors, len(names))):
                factor_names[f"Factor_{i+1}"] = names[i]
        
        except Exception as e:
            print(f"⚠️  LLM 命名失敗: {e}，使用預設名稱")
        
        # 補充缺失的因素
        for i in range(n_factors):
            if f"Factor_{i+1}" not in factor_names:
                factor_names[f"Factor_{i+1}"] = f"Factor_{i+1}"
        
        return factor_names
    
    def _assess_factor_quality(self, cfa_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        評估因素品質（因素負荷量、信度等）
        """
        loadings_df = cfa_results['loadings_df']
        n_factors = cfa_results['n_factors']
        
        assessment = {
            "overall_quality": "Good",
            "reliability": {
                "cronbach_alpha": cfa_results['cronbach_alpha'],
                "interpretation": self._interpret_alpha(cfa_results['cronbach_alpha'])
            },
            "loading_quality": {}
        }
        
        # 檢查每個因素的負荷量品質
        for factor in loadings_df.columns:
            factor_loadings = loadings_df[factor].abs()
            high_loadings = (factor_loadings >= 0.5).sum()
            mean_loading = factor_loadings.mean()
            min_loading = factor_loadings.min()
            
            assessment["loading_quality"][factor] = {
                "high_loading_items": int(high_loadings),
                "mean_loading": float(mean_loading),
                "min_loading": float(min_loading),
                "quality": "Excellent" if mean_loading >= 0.6 else "Good" if mean_loading >= 0.5 else "Fair"
            }
        
        # 整體評價
        mean_alpha = cfa_results['cronbach_alpha']
        if mean_alpha >= 0.8:
            assessment["overall_quality"] = "Excellent"
        elif mean_alpha >= 0.7:
            assessment["overall_quality"] = "Good"
        elif mean_alpha >= 0.6:
            assessment["overall_quality"] = "Fair"
        else:
            assessment["overall_quality"] = "Poor"
        
        return assessment
    
    def _interpret_alpha(self, alpha: float) -> str:
        """
        解釋 Cronbach's Alpha 值
        """
        if alpha >= 0.9:
            return "優異 (α ≥ 0.9)"
        elif alpha >= 0.8:
            return "良好 (0.8 ≤ α < 0.9)"
        elif alpha >= 0.7:
            return "可接受 (0.7 ≤ α < 0.8)"
        elif alpha >= 0.6:
            return "勉強可接受 (0.6 ≤ α < 0.7)"
        else:
            return "不可接受 (α < 0.6)，需要改進"
    
    def _generate_item_suggestions(self,
                                   cfa_results: Dict[str, Any],
                                   factor_names: Dict[str, str]) -> Dict[str, List[str]]:
        """
        根據因素負荷量生成題目改進建議
        """
        loadings_df = cfa_results['loadings_df']
        suggestions = {
            "items_to_remove": [],
            "items_to_review": [],
            "items_to_keep": []
        }
        
        for item in loadings_df.index:
            max_loading = loadings_df.loc[item].abs().max()
            
            if max_loading < 0.3:
                suggestions["items_to_remove"].append(f"{item} (最高負荷量: {max_loading:.3f})")
            elif max_loading < 0.5:
                suggestions["items_to_review"].append(f"{item} (最高負荷量: {max_loading:.3f})")
            else:
                suggestions["items_to_keep"].append(f"{item} (最高負荷量: {max_loading:.3f})")
        
        return suggestions
    
    def _generate_comprehensive_report(self,
                                       theoretical_background: str,
                                       cfa_results: Dict[str, Any],
                                       factor_names: Dict[str, str],
                                       quality_assessment: Dict[str, Any],
                                       item_suggestions: Dict[str, List[str]],
                                       n_factors: int) -> str:
        """
        生成完整的分析報告
        """
        report = f"""
{'='*70}
📊 驗證性因素分析 (CFA) 詳細報告
{'='*70}

🎯 理論背景
{'-'*70}
{theoretical_background}

📈 因素結構
{'-'*70}
因素數量: {n_factors}
因素名稱: {', '.join(factor_names.values())}

📋 信度分析
{'-'*70}
Cronbach's Alpha: {quality_assessment['reliability']['cronbach_alpha']:.4f}
解釋: {quality_assessment['reliability']['interpretation']}
整體品質評級: {quality_assessment['overall_quality']}

🔢 因素負荷量分析
{'-'*70}
{cfa_results['loadings_df'].round(3).to_string()}

📊 解釋方差比例
{'-'*70}
{', '.join([f'Factor_{i+1}: {var:.1%}' for i, var in enumerate(cfa_results['explained_variance'])])}

⭐ 因素負荷量品質評估
{'-'*70}
"""
        
        for factor, quality in quality_assessment['loading_quality'].items():
            report += f"\n{factor}:"
            report += f"\n  - 高負荷量題目數: {quality['high_loading_items']}"
            report += f"\n  - 平均負荷量: {quality['mean_loading']:.3f}"
            report += f"\n  - 品質評級: {quality['quality']}"
        
        report += f"\n\n✏️  題目改進建議\n{'-'*70}\n"
        
        if cfa_results['loadings_df'].shape[1] < cfa_results['loadings_df'].shape[0]:
            report += f"\n🗑️  建議刪除的題目 ({len(item_suggestions['items_to_remove'])} 個):\n"
            for item in item_suggestions['items_to_remove'][:10]:  # 最多顯示 10 個
                report += f"  • {item}\n"
            if len(item_suggestions['items_to_remove']) > 10:
                report += f"  ... 共 {len(item_suggestions['items_to_remove'])} 個\n"
            
            report += f"\n⚠️  建議審視的題目 ({len(item_suggestions['items_to_review'])} 個):\n"
            for item in item_suggestions['items_to_review'][:10]:
                report += f"  • {item}\n"
            if len(item_suggestions['items_to_review']) > 10:
                report += f"  ... 共 {len(item_suggestions['items_to_review'])} 個\n"
            
            report += f"\n✅ 保留的題目 ({len(item_suggestions['items_to_keep'])} 個):\n"
            for item in item_suggestions['items_to_keep'][:10]:
                report += f"  • {item}\n"
            if len(item_suggestions['items_to_keep']) > 10:
                report += f"  ... 共 {len(item_suggestions['items_to_keep'])} 個\n"
        
        report += f"\n\n💡 建議\n{'-'*70}\n"
        
        alpha = quality_assessment['reliability']['cronbach_alpha']
        if alpha >= 0.8:
            report += "✅ 測驗信度良好，因素結構穩定，可以考慮進一步驗證\n"
        elif alpha >= 0.7:
            report += "⚠️  測驗信度尚可，建議刪除或修改低負荷量的題目\n"
        else:
            report += "❌ 測驗信度偏低，強烈建議進行題目修訂或因素結構調整\n"
        
        report += f"\n{'='*70}\n"
        
        return report
