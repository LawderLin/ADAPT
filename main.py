from LM_AIG.System import LM_AIG_System

specifications = """
請設計一個「Brainrot 影響程度量表」（Brainrot Impact Scale），用於評估個人因長期接觸低品質數位內容（如短影片、碎片資訊）而產生的腦腐影響。

核心構念：
- 認知衰退：專注力下降、深度思考減弱。
- 數位成癮：過度分心、拖延行為。
- 媒體過載：忽略深度內容、依賴短暫刺激。

要求：
1. 以第一人稱撰寫。
2. 使用 Likert 量表題目。
3. 確保題項可靠、多面向，涵蓋日常情境（如滑手機、Doomscrolling）。
4. 避免雙重否定和模糊語言，確保題目清晰易懂。
"""

def main():
    # 建立完整系統
    lm_aig_system = LM_AIG_System()
    print("✅ LM-AIG 完整系統已初始化！")

    # 執行完整工作流程
    results = lm_aig_system.run_complete_workflow(
        specifications=specifications,
        num_items=10,
        max_iterations=3
    )

    # 顯示最終結果
    print("\n" + "="*40)
    print("🚀 LM-AIG 工作流程完成！")
    print("="*40)

    print("\n📝 最終題目列表:")
    for idx, item in enumerate(results["final_items"], 1):
        print(f"  {idx}. {item}")

    print("\n🔄 各迭代評分:")
    for iteration in results["iterations"]:
        score = iteration.get("overall_score", 0)
        print(f"  第 {iteration['iteration']} 次迭代: {score}/10")

if __name__ == "__main__":
    main()