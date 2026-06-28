import openai

def classify_question(question, api_key):
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""You are an expert video understanding assistant. Your task is to analyze the **Temporal Granularity** required to answer the user's question.
    Output a continuous float value between 0.0 and 1.0, representing the degree of "Short-term vs. Long-term" focus.
    Definitions and reference anchors (You may output ANY float value in between):
    
    Short Term Focus (→ 1.0): 
    - Focuses on immediate details, specific actions, or instantaneous events.
    - Emphasizes a quick understanding of a specific scene or object.
    - Can usually be answered using a single frame or frames from very closely spaced time intervals.
    - Keywords: Visual attributes, object counting, OCR, momentary micro-actions.

    Long Term Focus (→ 0.0): 
    - Focuses on overall plot development, character changes, or relationships spanning a long period.
    - Emphasizes a comprehensive understanding of the video's overall content and themes.
    - Requires combining frame information from multiple distant time periods.
    - Keywords: Causality ("Why"), Plot summary, Evolution, Abstract themes.

Please generate the degree value of "Short-term vs. Long-term" for this query.
Question: {question}
Provide your response in the form of a float value between 0.0 and 1.0:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()

    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content