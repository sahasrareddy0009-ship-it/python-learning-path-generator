import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_phase(phase_name: str, topics: list, level: str, goal: str) -> str:
    prompt = f"""
You are an expert Python mentor.

Student level: {level}
Career goal: {goal}

Learning phase: {phase_name}
Topics: {", ".join(topics)}

Explain WHY this phase is important for the student,
in simple, motivating language (3–4 lines).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )

    return response.choices[0].message.content.strip()
