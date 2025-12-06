import datetime
from LM_AIG.System import LM_AIG_System

prompts = [
"""
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
""", 
"""
AI use anxiety is a growing concern among workers as artificial intelligence becomes more prevalent in the workplace. Many employees feel intimidated or apprehensive about using AI tools in their daily work tasks. Here are some key points about AI use anxiety:
## Prevalence of AI Anxiety
Recent surveys indicate that AI anxiety is widespread:
- 71% of employees are concerned about AI negatively impacting their job security
- 38% of workers fear AI may make some or all of their job duties obsolete
- 48% of employees are more concerned about AI now than they were a year ago
## Causes of AI Use Anxiety
Several factors contribute to employees feeling anxious about using Al at work:
- Lack of training and skills: 73% worry they won't have opportunities to learn skills
- Concerns about job obsolescence: 75% fear AI will make certain jobs obsolete
- Uncertainty about AI capabilities: Many are unsure how to effectively use AI tools
- Fear of making mistakes: Workers worry about using AI incorrectly
## Impact on Mental Health and Performance
AI anxiety can have significant negative effects:
- 51% of those worried about AI say it negatively impacts their mental health
- 66% of AI-anxious workers report burnout, compared to 40% of workers overall
- Workers anxious about AI are 27% less likely to stay with their employer 
## Strategies to Address AI Use Anxiety

Employers can take several steps to ease anxiety and support employees:
- Provide comprehensive AI training and upskilling opportunities
- Communicate transparently about how AI will be used in the organization 
- Position AI as augmenting human work rather than replacing it 
- Involve employees in AI implementation processes
- Designate "Al champions" to support colleagues in using new tools 
- Offer mental health resources to help employees cope with anxiety

By addressing AI use anxiety proactively, organizations can help employees feel more comfortable and confident in leveraging AI technologies effectively in their work. This is crucial for successful AI adoption and maintaining a positive workplace culture.
"""
]

specifications = prompts[1]

def run_LM_AIG_workflow():
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
    for idx, item in enumerate(results.get("final_items", []), 1):
        print(f"  {idx}. {item}")

    print("\n🔄 各迭代評分:")
    for iteration in results.get("iterations", []):
        score = iteration.get("overall_score", 0)
        print(f"  第 {iteration['iteration']} 次迭代: {score}/10")

    # 將結果儲存為文字檔
    with open(f"lm_aig_workflow_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
        f.write("LM-AIG 工作流程結果\n")
        f.write("="*40 + "\n\n")
        f.write("最終題目列表:\n")
        for idx, item in enumerate(results["final_items"], 1):
            f.write(f"{idx}. {item}\n")
        f.write("\n各迭代評分:\n")
        for iteration in results["iterations"]:
            score = iteration.get("overall_score", 0)
            f.write(f"第 {iteration['iteration']} 次迭代: {score}/10\n")
    print(f"\n✅ 結果已儲存至 lm_aig_workflow_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def run_data_analysis_workflow():
    pass

if __name__ == "__main__":
    run_LM_AIG_workflow()
    # run_data_analysis_workflow()