import json
import os
import sys
import threading
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import sounddevice as sd
from dotenv import load_dotenv

from vosk import KaldiRecognizer, Model

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AUDIO_PATH = AUDIO_DIR / "question.wav"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_SECONDS = 7.0
DEFAULT_CHANNELS = 1


@lru_cache(maxsize=1)
def _load_vosk_model(model_dir_str: str) -> Model:
    model_dir = Path(model_dir_str)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Vosk model folder not found: {model_dir}\n"
            f"Put the unpacked model here or set VOSK_MODEL_PATH."
        )
    return Model(str(model_dir))


def get_vosk_model(model_path: str | Path | None = None) -> Model:
    if model_path is None:
        model_path = os.getenv(
            "VOSK_MODEL_PATH",
            str(BASE_DIR / "models" / "vosk-model-en-in"),
        )
    return _load_vosk_model(str(Path(model_path).resolve()))


def record_audio(
    output_path: str | Path = DEFAULT_AUDIO_PATH,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int | None = None,
) -> tuple[Path, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[bytes] = []
    stop_event = threading.Event()

    def wait_for_enter_to_stop() -> None:
        try:
            input()
        finally:
            stop_event.set()

    def callback(indata, frame_count, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[speech] audio status: {status}", file=sys.stderr)
        frames.append(bytes(indata))

    print("Speak now. Press Enter again to stop early.")
    stopper = threading.Thread(target=wait_for_enter_to_stop, daemon=True)
    stopper.start()

    start_time = time.monotonic()
    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=8000,
            device=device,
            channels=DEFAULT_CHANNELS,
            dtype="int16",
            callback=callback,
        ):
            while not stop_event.is_set():
                if (time.monotonic() - start_time) >= max_seconds:
                    break
                time.sleep(0.05)
    except Exception as exc:
        raise RuntimeError(f"Recording failed: {exc}") from exc

    duration = time.monotonic() - start_time

    audio_bytes = b"".join(frames)
    if not audio_bytes:
        raise RuntimeError("No audio captured from microphone.")

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(DEFAULT_CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    return output_path, duration


def speech_to_text(
    audio_path: str | Path,
    model_path: str | Path | None = None,
) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_vosk_model(model_path)

    with wave.open(str(audio_path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Vosk expects mono WAV audio.")
        if wf.getsampwidth() != 2:
            raise ValueError("Vosk expects 16-bit PCM WAV audio.")

        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(False)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

        final_result = json.loads(recognizer.FinalResult())
        return str(final_result.get("text", "")).strip()


def confirm_text(text: str) -> bool:
    print("\nI heard:")
    print(f'"{text}"')
    while True:
        choice = input("[Y] Confirm  [R] Retry: ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"r", "retry", "n", "no"}:
            return False
        print("Please enter Y or R.")


def listen(
    output_path: str | Path = DEFAULT_AUDIO_PATH,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    input("Press Enter to start recording...")

    try:
        audio_path, duration = record_audio(
            output_path=output_path,
            max_seconds=max_seconds,
            sample_rate=sample_rate,
            device=device,
        )
        text = speech_to_text(audio_path, model_path=model_path)

        if not text:
            return {
                "success": False,
                "retry": True,
                "text": "",
                "duration": duration,
                "audio_path": str(audio_path),
                "error": "No speech recognized.",
            }

        confirmed = confirm_text(text)
        return {
            "success": confirmed,
            "retry": not confirmed,
            "text": text if confirmed else "",
            "duration": duration,
            "audio_path": str(audio_path),
            "error": "" if confirmed else "User chose retry.",
        }

    except Exception as exc:
        print(f"[speech] {exc}", file=sys.stderr)
        return {
            "success": False,
            "retry": True,
            "text": "",
            "duration": 0.0,
            "audio_path": "",
            "error": str(exc),
        }


if __name__ == "__main__":
    result = listen()
    print(result)