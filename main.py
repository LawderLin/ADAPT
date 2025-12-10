from LM_AIG.System import LM_AIG_System

prompts = ["""
在中庸思維的體系下，個人所著重的是自我在不同環境中的權宜表現，而不是跨情境的道德標準的衡量與評價。據此，研究者整理出個人在不同環境中權宜表現的準則，其一為「權」，也就是認清外在的訊息與自己本身的內在要求，並詳加考慮，其所隱含的思維特質研究者將之命名為「多方思考」；再者為「和」，此特色包括兩個方面，其一為個人整合外在環境的訊息與內在個體的想法，其二為以不偏激以及和諧的方式做為行動的準則，此二特色所隱含的思維特質分別為 「整合性」 與 「和諧性」。

請根據上述背景，設計「中庸思維量表」之「和諧性」子量表，包含 **5** 題 Likert 量表題目。不需要其他子構念之題目。
"""]

def run_LM_AIG_workflow(specifications: str = prompts[0], max_iterations: int = 3, num_items: int = 10):
    # 建立完整系統
    lm_aig_system = LM_AIG_System()
    print("✅ LM-AIG 完整系統已初始化！")

    # 執行完整工作流程
    result = lm_aig_system.run_complete_workflow(
        specifications=specifications,
        num_items=num_items,
        max_iterations=max_iterations
    )

def run_data_analysis_workflow():
    pass

if __name__ == "__main__":
    max_iterations = 5
    num_items = 5
    run_LM_AIG_workflow( 
        max_iterations=max_iterations, 
        num_items=num_items)