import ollama
import re
import json

from config import config

class ConstructSettingAgent:
    """構念設定代理人：負責定義心理學構念"""

    def __init__(self):
        self.model = config.model_name
        self.language = config.language

    def draft_construct_definition(self, topic: str):
        """
        根據主題草擬構念定義

        :param topic: 心理構念主題
        :returns str: 構念定義的詳細描述，傳給 ItemWriterAgent 使用
        """
        prompt = f"""
        You are an expert in psychological construct development. Please help draft a detailed definition of the psychological construct based on the following specifications: 
        ---
        {topic}
        ---
        Please provide:
        1. **Operational Definition**: Precisely describe what this construct is.
        2. **Main Dimensions**: Into which dimensions can this construct be divided? (Recommend 3-5)
        3. **Inclusion/Exclusion Criteria**: What belongs to this construct, and what does not?
        4. **Recommended Target Participants**.

        Respond with only the construct definition list. Do not include any additional explanations.
        The definition should be in {self.language}.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.7}
            )

            content = response['message']['content']
            return content

        except Exception as e:
            return {"error": str(e)}
