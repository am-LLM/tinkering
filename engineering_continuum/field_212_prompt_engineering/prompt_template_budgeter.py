"""Course 212: Prompt Template Formatter & Token Budget Truncation Manager"""
class PromptTemplateBudgeter:
    @staticmethod
    def format_with_budget(template: str, variables: dict, max_chars: int = 1000) -> str:
        prompt = template
        for k, v in variables.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))
        if len(prompt) > max_chars:
            return prompt[:max_chars - 3] + "..."
        return prompt
