import openai
import ast
import os
import re

def classify_question(question, candidates, api_key=None):
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=os.getenv("REMEM_LLM_BASE_URL") or None)
    
    prompt = f"""
You are an expert in visual perception and long-term knowledge reasoning.
Task: Generate 3-5 high-quality "Visual Entities" (keywords) to help a video model locate the specific frames needed to distinguish between the given candidates.

Question: {question}
Candidates: {candidates}

Strategic Reasoning Instructions:
1. Identify the 'Core Objects' mentioned in the candidates.
2. For each object, translate it into its 'Visual Signature' (shape, color, or distribution) based on common world knowledge. 
3. Focus on 'Discriminative Features': What visual details would confirm one candidate over the others?
4. Ensure the keywords cover all potential evidence needed to answer the question.

Output Requirement:
- Return ONLY a Python list of strings.
- Each keyword should be a short, visually descriptive phrase.
- DO NOT provide any conversation or explanation.

Output:
"""
    response = client.chat.completions.create(
        model=os.getenv("REMEM_LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}]
        ,temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    
    content = response.choices[0].message.content.strip()
    
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "".join(lines).strip()

    try:
        memory_list = ast.literal_eval(content)
        if isinstance(memory_list, list):
            return memory_list
        else:
            return [content] 
    except Exception as e:
        print(f"Error parsing entity context: {e}")
        # Some local models omit escaping apostrophes inside Python-quoted items.
        items = re.split(r"\\?['\"]\s*,\s*\\?['\"]", content.strip("[] \n"))
        return [item.strip(" '\"\n") for item in items if item.strip(" '\"\n")]
