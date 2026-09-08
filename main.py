import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sounddevice as sd
from dotenv import load_dotenv

import expert
import speech
import tts
import vision


# ==========================================================
# Setup
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MAX_HISTORY = 5
MAX_STT_FAILURES = 3
DEFAULT_IMAGE_PATH = BASE_DIR / "images" / "sample2.jpg"
LOG_DIR = BASE_DIR / "logs"

EXIT_KEYWORDS = {
    "exit",
    "quit",
    "stop",
    "bye",
    "goodbye",
    "no more",
    "that's all",
    "that is all",
    "done",
}


# ==========================================================
# Logging
# ==========================================================

def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info("Log file: %s", log_file)


# ==========================================================
# Helpers
# ==========================================================

def resolve_image_path(arg: str | None) -> Path:
    if not arg:
        return DEFAULT_IMAGE_PATH
    p = Path(arg)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def safe_speak(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True

    try:
        ok = tts.speak(text, wait=True)
        if not ok:
            logging.warning("TTS returned failure.")
            print(text)
        return ok
    except Exception:
        logging.exception("TTS failed.")
        print(text)
        return False


def build_session(art_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "art": art_context,
        "history": [],
        "question_count": 0,
        "active": True,
    }


def build_intro_message(art_context: dict[str, Any]) -> str:
    art_style = art_context.get("art_style", "Unknown")
    origin = art_context.get("origin", "Unknown")
    confidence = float(art_context.get("confidence", 0.0))
    needs_review = bool(art_context.get("needs_review", False))

    if art_style == "Unknown" or confidence < 0.75 or needs_review:
        return (
            f"I think this artwork may be {art_style} from {origin}, "
            f"but I am not fully certain."
        )

    return f"I identified this artwork as {art_style} from {origin}."


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_exit_command(text: str) -> bool:
    t = normalize_text(text)

    if t in {"no", "nope", "nah", "none", "nothing", "never"}:
        return True

    patterns = [
        r"\bno\b",
        r"\bno thanks\b",
        r"\bno thank you\b",
        r"\bno more\b",
        r"\bnothing else\b",
        r"\bthat's all\b",
        r"\bthat is all\b",
        r"\bdone\b",
        r"\bstop\b",
        r"\bexit\b",
        r"\bquit\b",
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bi do not have any (more )?(questions?|doubts?|queries?)\b",
        r"\bi don't have any (more )?(questions?|doubts?|queries?)\b",
        r"\bi have no (more )?(questions?|doubts?|queries?)\b",
        r"\bno (more )?(questions?|doubts?|queries?)\b",
    ]

    return any(re.search(p, t) for p in patterns)


def append_history(session: dict[str, Any], user_question: str, answer: str) -> None:
    session["history"].append(
        {
            "user": user_question,
            "assistant": answer,
        }
    )
    session["history"] = session["history"][-MAX_HISTORY:]
    session["question_count"] += 1


def startup_check(image_path: Path | None = None) -> None:
    logging.info("Running startup checks...")

    required_files = [
        BASE_DIR / "prompt.txt",
        BASE_DIR / "expert_prompt.txt",
    ]
    if image_path is not None:
        required_files.append(image_path)

    for item in required_files:
        if not item.exists():
            raise FileNotFoundError(f"Missing required file: {item}")
        logging.info("Found: %s", item)

    logging.info("Checking offline speech model...")
    speech.get_vosk_model()
    logging.info("Vosk model loaded.")

    logging.info("Checking Piper TTS...")
    tts.init_tts()
    logging.info("Piper TTS ready.")

    try:
        devices = sd.query_devices()
        has_input = any(
            dev.get("max_input_channels", 0) > 0
            for dev in devices
        )
        if has_input:
            logging.info("Microphone detected.")
        else:
            logging.warning("No input microphone detected by sounddevice.")
    except Exception as exc:
        logging.warning("Microphone check could not be completed: %s", exc)

    logging.info("Startup checks complete.")


def compact_text(text: str, max_words: int = 24) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def make_spoken_response(response: dict[str, Any]) -> str:
    """
    Build a short spoken response from structured Gemini output.
    Keeps TTS concise while still covering the key facts.
    """
    parts: list[str] = []

    summary = str(response.get("summary", "")).strip()
    origin = str(response.get("origin", "")).strip()
    why_famous = str(response.get("why_famous", "")).strip()
    cultural_impact = str(response.get("cultural_impact", "")).strip()
    key_feature = str(response.get("key_feature", "")).strip()
    notable_person = str(response.get("notable_person", "")).strip()

    if summary:
        parts.append(compact_text(summary, 28))
    else:
        if origin:
            parts.append(f"It comes from {origin}.")
        if why_famous:
            parts.append(compact_text(f"It is known because {why_famous}", 18))

    # Add only a few extra facts so speech stays short.
    if origin and origin not in summary:
        parts.append(f"It comes from {origin}.")
    if why_famous:
        parts.append(compact_text(f"It became famous because {why_famous}", 18))
    if cultural_impact:
        parts.append(compact_text(f"Its cultural impact is {cultural_impact}", 18))
    if key_feature:
        parts.append(compact_text(f"A key feature is {key_feature}", 18))
    if notable_person:
        parts.append(compact_text(f"A notable person linked to it is {notable_person}", 18))

    # Keep it from becoming too long.
    final_parts: list[str] = []
    for p in parts:
        if p and p not in final_parts:
            final_parts.append(p)
        if len(final_parts) >= 4:
            break

    return " ".join(final_parts).strip()


def print_structured_response(response: dict[str, Any]) -> None:
    print("\n========== ANSWER ==========\n")
    for key in [
        "summary",
        "origin",
        "why_famous",
        "cultural_impact",
        "key_feature",
        "notable_person",
        "follow_up",
    ]:
        value = str(response.get(key, "")).strip()
        if value:
            label = key.replace("_", " ").upper()
            print(f"{label}: {value}\n")
    print("============================\n")


def speech_result_to_text(result: dict[str, Any]) -> str:
    return (result.get("text") or "").strip()


def ask_expert_with_retry(art, question, history):
    """
    Calls Gemini with automatic retry.
    Returns:
        dict -> success
        None -> failed after all retries
    """
    max_retries = 3
    spoke_wait = False

    for attempt in range(max_retries):
        try:
            logging.info(
                "Gemini attempt %d/%d",
                attempt + 1,
                max_retries,
            )
            return expert.ask_expert(
                art,
                question,
                history,
            )

        except Exception as e:
            error = str(e).lower()
            logging.warning("Gemini Error: %s", e)

            temporary = any(
                x in error
                for x in [
                    "503",
                    "500",
                    "429",
                    "timeout",
                    "connection",
                    "remoteprotocol",
                    "unavailable",
                    "reset",
                    "json",
                ]
            )

            if temporary and attempt < max_retries - 1:
                if not spoke_wait:
                    safe_speak("One moment please.")
                    spoke_wait = True

                wait = 2 ** attempt
                logging.info("Retrying in %d seconds...", wait)
                time.sleep(wait)
                continue

            logging.error("Gemini failed permanently.")
            return None

    return None


def prepare_initial_art_context(image_arg: str | None = None) -> tuple[Path, dict[str, Any]]:
    """
    Capture from the webcam when no image path is provided.
    This is the only necessary change for the new camera flow.
    """
    if image_arg:
        image_path = resolve_image_path(image_arg)
        startup_check(image_path)
        logging.info("Identifying artwork from: %s", image_path)
        art_context = vision.identify_art(image_path)
        return image_path, art_context

    startup_check(None)

    logging.info("Capturing artwork from webcam...")
    capture = vision.capture_artwork(
        preview_seconds=4.0,
        show_preview=True,
        auto_retry=True,
    )

    if not capture["success"]:
        raise RuntimeError(
            f"Webcam capture failed: {capture['reason']}"
        )

    image_path = Path(capture["image_path"])
    logging.info("Captured artwork image: %s", image_path)
    logging.info("Capture reason: %s", capture["reason"])
    if capture.get("metrics"):
        logging.info("Capture metrics: %s", capture["metrics"])

    logging.info("Identifying artwork from webcam capture...")
    art_context = vision.identify_art(image_path)
    return image_path, art_context


# ==========================================================
# Conversation Loop
# ==========================================================

def run_conversation(session: dict[str, Any]) -> None:
    intro = build_intro_message(session["art"])
    safe_speak(intro)
    safe_speak("Do you have any questions?")

    stt_failures = 0

    while session["active"]:
        result = speech.listen()

        if result.get("retry"):
            continue

        if not result.get("success"):
            stt_failures += 1
            logging.warning("Speech input failed: %s", result.get("error", ""))

            if stt_failures >= MAX_STT_FAILURES:
                safe_speak("I'm having trouble hearing right now. Let's pause here.")
                break

            safe_speak("I didn't catch that. Please try again.")
            continue

        stt_failures = 0

        user_question = speech_result_to_text(result)
        if not user_question:
            safe_speak("Please ask your question again.")
            continue

        logging.info("User question: %s", user_question)

        if is_exit_command(user_question):
            safe_speak("Okay. Ending the session.")
            session["active"] = False
            break

        response = ask_expert_with_retry(
            session["art"],
            user_question,
            session["history"],
        )

        if response is None:
            safe_speak(
                "Sorry, I'm having trouble reaching the knowledge server right now. Please ask that question again."
            )
            continue

        if not isinstance(response, dict):
            logging.warning("Expert returned non-dict response: %r", response)
            safe_speak("Sorry, I couldn't answer that right now.")
            continue

        print_structured_response(response)

        spoken_answer = make_spoken_response(response)
        if not spoken_answer:
            spoken_answer = str(response.get("summary", "")).strip()

        logging.info("Answer %d: %s", session["question_count"] + 1, spoken_answer)

        safe_speak(spoken_answer)

        append_history(session, user_question, spoken_answer)

        follow_up = str(response.get("follow_up", "")).strip()
        if follow_up:
            safe_speak(follow_up)
        else:
            safe_speak("Do you have another question?")


# ==========================================================
# Main
# ==========================================================

def main() -> int:
    setup_logging()

    image_arg = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        _, art_context = prepare_initial_art_context(image_arg)

        print("\n========== ART CONTEXT ==========")
        print()
        print(json.dumps(art_context, indent=2, ensure_ascii=False))

        session = build_session(art_context)
        run_conversation(session)

        return 0

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        safe_speak("Session ended.")
        return 0

    except Exception as exc:
        logging.exception("Fatal error.")
        print(f"\nFatal error: {exc}")
        return 1

    finally:
        try:
            tts.shutdown()
        except Exception:
            logging.exception("TTS shutdown failed.")


if __name__ == "__main__":
    raise SystemExit(main())