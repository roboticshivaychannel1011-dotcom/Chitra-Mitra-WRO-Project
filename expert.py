import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

# ==========================================================
# Setup
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gemini-flash-latest")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=API_KEY)


# ==========================================================
# Helpers
# ==========================================================

def read_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_context(context: dict) -> str:
    features = context.get("key_features", [])

    if isinstance(features, list):
        features_text = "\n".join(f"- {f}" for f in features) if features else "- None"
    else:
        features_text = f"- {features}"

    return f"""
Artwork Style: {context.get("art_style", "Unknown")}
Origin: {context.get("origin", "Unknown")}
Confidence: {context.get("confidence", 0.0)}
Needs Review: {context.get("needs_review", False)}

Visible Features:
{features_text}
""".strip()


def build_history(history: list | None) -> str:
    if not history:
        return "No previous conversation."

    text = []
    for item in history[-5:]:
        text.append(f"User: {item['user']}")
        text.append(f"Assistant: {item['assistant']}")
        text.append("")

    return "\n".join(text).strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group(0))


# ==========================================================
# Main
# ==========================================================

def ask_expert(
    art_context: dict,
    user_question: str,
    conversation_history: list | None = None,
) -> dict[str, Any]:
    """
    Return compact structured facts for the identified artwork.
    The returned dict is meant to be consumed by main.py.
    """
    expert_prompt = read_text(BASE_DIR / "expert_prompt.txt")

    prompt = f"""
{expert_prompt}

==================================================
ARTWORK CONTEXT
==================================================

{build_context(art_context)}

==================================================
PREVIOUS CONVERSATION
==================================================

{build_history(conversation_history)}

==================================================
CURRENT USER QUESTION
==================================================

{user_question}

==================================================
TASK
==================================================

Return ONLY valid JSON in this exact structure:

{{
  "summary": "",
  "origin": "",
  "why_famous": "",
  "cultural_impact": "",
  "key_feature": "",
  "notable_person": "",
  "follow_up_prompt": "",
  "follow_up_topic": ""
}}

Rules:
- Keep summary short and natural for TTS.
- Use simple spoken English.
- Do not write long paragraphs.
- Keep each field compact.
- If a field is unknown or not relevant, use an empty string.
- Do not include markdown, code fences, or extra text.
- Do not identify the artwork again.
- Use the supplied art context and conversation history.
- Make follow_up_prompt dynamic and closely related to the current answer.
- Make follow_up_topic a short reusable topic phrase for the next turn.
- If no good follow-up is needed, set both follow_up_prompt and follow_up_topic to empty strings.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return extract_json(response.text or "")


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":
    sample_context = {
        "art_style": "Warli",
        "confidence": 0.97,
        "origin": "Maharashtra, India",
        "key_features": [
            "White pigment",
            "Triangle human figures",
            "Community dance",
        ],
        "needs_review": False,
    }

    history = [
        {
            "user": "Tell me about this art.",
            "assistant": "Warli art is a tribal art form...",
        }
    ]

    question = input("Question: ")
    answer = ask_expert(sample_context, question, history)

    print("\n========== JSON ANSWER ==========\n")
    print(json.dumps(answer, indent=2, ensure_ascii=False))