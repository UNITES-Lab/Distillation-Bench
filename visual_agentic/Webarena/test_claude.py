import litellm

DEFAULT_SYSTEM_PROMPT = """You are an AI assistant. Your goal is to provide informative and substantive responses to queries."""


def call_claude(
    prompt,
    model_id="claude-3-5-sonnet-20240620",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    response_text = litellm.completion(
        model=model_id,
        max_tokens=1024,
        temperature=0,  # 0.95,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response_text["choices"][0]["message"]["content"]


print(call_claude("Hello, how are you?")["choices"][0]["message"]["content"])
