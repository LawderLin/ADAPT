import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

class DataAnalysisPipeline:
    """
    資料分析管線，用於分析測驗題目的心理計量特性
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.factor_analysis = None
        self.data = None
        self.scaled_data = None

    def load_simulated_data(self, n_participants: int = 200, n_items: int = 10) -> pd.DataFrame:
        """
        生成模擬測驗資料（實際使用時應載入真實資料）

        Args:
            n_participants: 受試者數量
            n_items: 題目數量

        Returns:
            模擬的測驗反應資料
        """
        # 生成模擬資料：假設有一個潛在因子
        np.random.seed(42)

        # 潛在因子分數
        latent_factor = np.random.normal(0, 1, n_participants)

        # 題目參數
        item_loadings = np.random.uniform(0.3, 0.8, n_items)
        item_difficulties = np.random.uniform(-2, 2, n_items)

        # 生成反應資料（使用 Rasch 模型概念但簡化為連續分數）
        responses = []
        for i in range(n_items):
            # 簡化的項目反應：因子負荷量 * 潛在因子 + 難度 + 隨機誤差
            item_scores = (item_loadings[i] * latent_factor +
                           item_difficulties[i] +
                           np.random.normal(0, 0.3, n_participants))
            # 轉換為 1-5 分的李克特量表
            item_scores = np.clip(np.round(3 + item_scores), 1, 5)
            responses.append(item_scores)

        # 建立 DataFrame
        data = pd.DataFrame(
            np.array(responses).T,
            columns=[f'Item_{i+1}' for i in range(n_items)]
        )

        self.data = data
        return data

    def exploratory_factor_analysis(self, n_factors: int = 2) -> Dict[str, Any]:
        """
        進行探索性因子分析 (EFA)

        Args:
            n_factors: 因子數量

        Returns:
            因子分析結果
        """
        if self.data is None:
            return {"error": "請先載入資料"}

        try:
            # 標準化資料
            self.scaled_data = self.scaler.fit_transform(self.data)

            # 因子分析
            self.factor_analysis = FactorAnalysis(
                n_components=n_factors, random_state=42)
            factor_scores = self.factor_analysis.fit_transform(
                self.scaled_data)

            # 因子負荷量
            loadings = self.factor_analysis.components_.T

            # 計算解釋變異量
            explained_variance = np.var(factor_scores, axis=0)
            explained_variance_ratio = explained_variance / \
                np.sum(explained_variance)

            # 計算項目間相關
            correlation_matrix = np.corrcoef(self.data.T)

            # 信度分析（Cronbach's Alpha）
            alpha = self._calculate_cronbach_alpha()

            results = {
                "n_factors": n_factors,
                "factor_loadings": loadings.tolist(),
                "factor_loadings_df": pd.DataFrame(
                    loadings,
                    columns=[f'Factor_{i+1}' for i in range(n_factors)],
                    index=self.data.columns
                ),
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "correlation_matrix": correlation_matrix.tolist(),
                "cronbach_alpha": alpha,
                "factor_scores": factor_scores,
                "item_statistics": self._calculate_item_statistics()
            }

            return results

        except Exception as e:
            return {"error": f"因子分析失敗: {str(e)}"}

    def _calculate_cronbach_alpha(self) -> float:
        """計算 Cronbach's Alpha 信度係數"""
        if self.data is None:
            return 0.0

        # 項目數量
        k = self.data.shape[1]

        # 項目變異數
        item_variances = self.data.var(axis=0, ddof=1)

        # 總分變異數
        total_variance = self.data.sum(axis=1).var(ddof=1)

        # Cronbach's Alpha 公式
        alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

        return alpha

    def _calculate_item_statistics(self) -> Dict[str, Any]:
        """計算題目統計量"""
        if self.data is None:
            return {}

        stats = {
            "means": self.data.mean().to_dict(),
            "std_devs": self.data.std().to_dict(),
            "skewness": self.data.skew().to_dict(),
            "kurtosis": self.data.kurtosis().to_dict(),
        }

        # 項目-總分相關
        total_scores = self.data.sum(axis=1)
        item_total_correlations = {}
        for col in self.data.columns:
            # 修正的項目-總分相關（排除該題目本身）
            corrected_total = total_scores - self.data[col]
            correlation = np.corrcoef(self.data[col], corrected_total)[0, 1]
            item_total_correlations[col] = correlation

        stats["item_total_correlations"] = item_total_correlations

        return stats

    def generate_analysis_report(self, efa_results: Dict[str, Any]) -> str:
        """生成分析報告"""
        if "error" in efa_results:
            return f"分析錯誤: {efa_results['error']}"

        report = f"""
📊 測驗心理計量分析報告
{'='*50}

🔢 基本資訊:
- 樣本數: {self.data.shape[0]}
- 題目數: {self.data.shape[1]}
- 因子數: {efa_results['n_factors']}

📈 信度分析:
- Cronbach's Alpha: {efa_results['cronbach_alpha']:.3f}

🎯 因子分析結果:
解釋變異量比例: {[f'{ratio:.3f}' for ratio in efa_results['explained_variance_ratio']]}

📋 因子負荷量:
{efa_results['factor_loadings_df'].round(3).to_string()}

📊 題目統計:
"""

        item_stats = efa_results['item_statistics']
        for item in self.data.columns:
            report += f"\n{item}:"
            report += f"  平均數: {item_stats['means'][item]:.2f}"
            report += f"  標準差: {item_stats['std_devs'][item]:.2f}"
            report += f"  項目-總分相關: {item_stats['item_total_correlations'][item]:.3f}"

        # 評估建議
        report += f"\n\n💡 評估建議:\n"

        if efa_results['cronbach_alpha'] >= 0.8:
            report += "✅ 信度良好 (α ≥ 0.8)\n"
        elif efa_results['cronbach_alpha'] >= 0.7:
            report += "⚠️ 信度尚可 (0.7 ≤ α < 0.8)，可考慮改進\n"
        else:
            report += "❌ 信度偏低 (α < 0.7)，建議重新檢視題目\n"

        # 檢查項目-總分相關
        low_correlation_items = [
            item for item, corr in item_stats['item_total_correlations'].items()
            if corr < 0.3
        ]

        if low_correlation_items:
            report += f"⚠️ 以下題目與總分相關偏低，建議檢視: {', '.join(low_correlation_items)}\n"
        else:
            report += "✅ 所有題目與總分相關良好\n"

        return report