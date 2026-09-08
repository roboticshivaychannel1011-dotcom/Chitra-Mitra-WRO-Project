import os
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
import winsound
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PIPER_BIN = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "").strip()

# Optional: set this in .env if you want a specific playback duration gap.
SENTENCE_PAUSE_SEC = float(os.getenv("TTS_SENTENCE_PAUSE_SEC", "0.05"))

# Internal state
_tts_queue: queue.Queue = queue.Queue()
_tts_thread: Optional[threading.Thread] = None
_tts_stop_event = threading.Event()
_tts_ready = False
_tts_lock = threading.Lock()


@dataclass
class SpeakJob:
    text: str
    speed: float
    wait: bool
    done: threading.Event
    ok: bool = False
    error: str = ""


def _ensure_dependencies() -> None:
    if not PIPER_MODEL_PATH:
        raise RuntimeError(
            "PIPER_MODEL_PATH is not set in .env. "
            "Point it to your Piper voice model .onnx file."
        )

    model_path = Path(PIPER_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Piper model not found: {model_path}")

    if shutil.which(PIPER_BIN) is None:
        raise RuntimeError(
            f"'{PIPER_BIN}' not found in PATH. Install Piper CLI and ensure it is accessible."
        )


def init_tts() -> None:
    global _tts_thread, _tts_ready

    with _tts_lock:
        if _tts_ready:
            return

        _ensure_dependencies()
        _tts_stop_event.clear()

        _tts_thread = threading.Thread(target=_worker_loop, daemon=True)
        _tts_thread.start()
        _tts_ready = True


def _split_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on sentence endings, but keep short phrases readable.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _run_piper(text: str, wav_path: Path, speed: float = 1.0) -> None:
    """
    Generates a WAV file using Piper.
    """
    # Piper uses length_scale; > 1.0 slower, < 1.0 faster.
    # Convert our friendly speed to Piper length_scale.
    length_scale = 1.0 / max(0.5, min(speed, 2.0))

    cmd = [
        PIPER_BIN,
        "--model",
        PIPER_MODEL_PATH,
        "--output_file",
        str(wav_path),
        "--length_scale",
        str(length_scale),
    ]

    proc = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            f"Piper synthesis failed.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )


def _play_wav(wav_path: Path) -> None:
    system = os.name.lower()

    if system == "nt":
        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        return

    # Linux / Raspberry Pi / macOS
    if shutil.which("aplay") is not None:
        subprocess.run(["aplay", str(wav_path)], check=True)
        return

    if shutil.which("afplay") is not None:
        subprocess.run(["afplay", str(wav_path)], check=True)
        return

    raise RuntimeError("No supported audio player found (aplay/afplay/winsound).")


def _worker_loop() -> None:
    while not _tts_stop_event.is_set():
        try:
            job: SpeakJob = _tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if _tts_stop_event.is_set():
                job.ok = False
                job.error = "TTS stopped."
                job.done.set()
                continue

            sentences = _split_text(job.text)
            if not sentences:
                job.ok = True
                job.done.set()
                continue

            for i, sentence in enumerate(sentences):
                if _tts_stop_event.is_set():
                    raise RuntimeError("TTS stopped.")

                wav_path = AUDIO_DIR / f"tts_{int(time.time() * 1000)}_{i}.wav"
                try:
                    _run_piper(sentence, wav_path, speed=job.speed)
                    _play_wav(wav_path)
                finally:
                    try:
                        if wav_path.exists():
                            wav_path.unlink()
                    except Exception:
                        pass

                if SENTENCE_PAUSE_SEC > 0:
                    time.sleep(SENTENCE_PAUSE_SEC)

            job.ok = True

        except Exception as exc:
            job.ok = False
            job.error = str(exc)

        finally:
            job.done.set()
            _tts_queue.task_done()


def speak(text: str, wait: bool = True, speed: float = 1.0) -> bool:
    """
    Speak text offline using Piper.
    - wait=True  -> block until speech is finished
    - wait=False -> queue speech and return immediately (Tkinter-friendly)
    """
    init_tts()

    text = (text or "").strip()
    if not text:
        return True

    done = threading.Event()
    job = SpeakJob(text=text, speed=speed, wait=wait, done=done)
    _tts_queue.put(job)

    if wait:
        done.wait()
        return job.ok

    return True


def stop() -> None:
    """
    Stop queued speech and prevent new output until re-initialized.
    """
    global _tts_ready

    _tts_stop_event.set()

    # Drain pending jobs
    try:
        while True:
            job = _tts_queue.get_nowait()
            if isinstance(job, SpeakJob):
                job.ok = False
                job.error = "TTS stopped."
                job.done.set()
            _tts_queue.task_done()
    except queue.Empty:
        pass

    _tts_ready = False


def shutdown() -> None:
    """
    Graceful cleanup.
    """
    stop()


if __name__ == "__main__":
    # Quick manual test
    ok = speak("Hello. I identified this as Warli art.", wait=True)
    print("TTS OK:", ok)