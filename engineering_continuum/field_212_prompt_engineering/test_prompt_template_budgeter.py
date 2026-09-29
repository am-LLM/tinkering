from prompt_template_budgeter import PromptTemplateBudgeter

def test_prompt_budgeter():
    tmpl = "Hello {name}, your score is {score}."
    res = PromptTemplateBudgeter.format_with_budget(tmpl, {"name": "Alice", "score": 95}, max_chars=100)
    assert res == "Hello Alice, your score is 95."
