import os
import sys
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageOps

# =========================
# Setup
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gemini-flash-latest")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


# =========================
# Helpers
# =========================
def read_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_image(image_path: Path, max_size=(768, 768)) -> Image.Image:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size)
    return img


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group(0))


def analyze_art(image_path: Path) -> dict:
    prompt = read_text(BASE_DIR / "prompt.txt")
    image = load_image(image_path)

    print("=" * 50)
    print("MODEL:", MODEL)
    print("Prompt length:", len(prompt))
    print("Image size:", image.size)
    print("=" * 50)

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image],
    )

    return extract_json(response.text or "")


# =========================
# Main
# =========================
def main():
    image_arg = sys.argv[1] if len(sys.argv) > 1 else "images/sample1.jpg"
    image_path = (BASE_DIR / image_arg).resolve() if not Path(image_arg).is_absolute() else Path(image_arg)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = analyze_art(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()