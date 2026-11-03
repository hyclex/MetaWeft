"""Small ReMe-derived executor; parent and deviations recorded in DESIGNS.md."""

import json
import re

from jinja2 import Template


def extract_code(text):
    match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    partial = re.match(r".*```python\n(.*)", text, flags=re.DOTALL)
    if partial:
        code = partial.group(1).strip()
        return code if code else text + ("" if text.endswith("\n") else "\n") + "```"
    return text


def execute(task_spec, guidance, *, llm, environment, prompt, max_interactions=40, record_step=None):
    system = Template(prompt.lstrip()).render(
        supervisor=task_spec["supervisor"],
        app_descriptions=json.dumps([{"name": k, "description": v}
                                     for k, v in task_spec["app_descriptions"].items()], indent=1),
    )
    query = task_spec["instruction"]
    if guidance:
        query += "\n\nTask-specific guidance from prior experience:\n" + guidance
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    trace = []
    for step in range(max_interactions):
        response = llm.chat(messages, phase="execution", component="react", step=step,
                            temperature=0.9, seed=0, extra_body={"enable_thinking": False})
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Executor returned no text")
        code = extract_code(text)
        messages.append({"role": "assistant", "content": code})
        observation = environment.execute(code)
        messages.append({"role": "user", "content": "Output:\n```\n" + observation + "```\n\n"})
        event = {"step": step, "code": code, "observation": observation}
        trace.append(event)
        if record_step:
            record_step(event)
        if environment.task_completed():
            break
    return trace
