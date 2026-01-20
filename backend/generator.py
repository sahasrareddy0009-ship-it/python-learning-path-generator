import json
from pathlib import Path
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge_base" / "python_paths.json"


def generate_learning_path(level: str, goal: str) -> str:
    with open(KB_PATH, "r") as f:
        knowledge = json.load(f)

    prompt = f"""
You are an experienced Python mentor.

Student level: {level}
Student goal: {goal}

Base curriculum:
{json.dumps(knowledge, indent=2)}

TASK:
- Select relevant topics
- Adapt depth based on level
- Explain WHY each phase matters
- Speak like a mentor, not a textbook
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content
