import os
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_PATH = CAPTURE_DIR / "artwork.jpg"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gemini-flash-latest")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

CAMERA_INDEX = 0
WARMUP_SECONDS = 1.0
BURST_FRAMES = 6
BURST_INTERVAL = 0.08
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 225.0
MIN_CONTRAST = 20.0
MIN_BLUR_SCORE = 70.0
MIN_CENTER_EDGE_DENSITY = 0.015


# ============================================================
# ORIGINAL GEMINI HELPERS
# ============================================================

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
        raise ValueError(f"No JSON found:\n{text}")

    return json.loads(match.group(0))


# ============================================================
# LOCAL IMAGE QUALITY
# ============================================================

def _quality_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h, w = gray.shape
    x1, x2 = int(w * 0.20), int(w * 0.80)
    y1, y2 = int(h * 0.15), int(h * 0.85)

    center = gray[y1:y2, x1:x2]
    edges = cv2.Canny(center, 80, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "center_edge_density": edge_density,
    }


def _quality_result(metrics):
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    blur_score = metrics["blur_score"]
    edge_density = metrics["center_edge_density"]

    if brightness < MIN_BRIGHTNESS:
        return {
            "ok": False,
            "reason": "The image is too dark. Move to a brighter area.",
            "metrics": metrics,
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "ok": False,
            "reason": "The image is too bright. Reduce glare or move away from direct light.",
            "metrics": metrics,
        }

    if contrast < MIN_CONTRAST:
        return {
            "ok": False,
            "reason": "The image has very low contrast. Please improve the lighting.",
            "metrics": metrics,
        }

    if blur_score < MIN_BLUR_SCORE:
        return {
            "ok": False,
            "reason": "The artwork looks blurry. Hold the camera steady.",
            "metrics": metrics,
        }

    if edge_density < MIN_CENTER_EDGE_DENSITY:
        return {
            "ok": False,
            "reason": "The artwork does not appear clearly inside the camera view.",
            "metrics": metrics,
        }

    return {
        "ok": True,
        "reason": "Image quality is good.",
        "metrics": metrics,
    }


def check_image_quality(frame):
    """
    Local-only quality check. Never contacts Gemini.
    """
    if frame is None:
        return {
            "ok": False,
            "reason": "No camera frame was captured.",
            "metrics": {},
        }

    return _quality_result(_quality_metrics(frame))


# ============================================================
# WEBCAM CAPTURE
# ============================================================

def _open_camera(camera_index=CAMERA_INDEX):
    if os.name == "nt":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(camera_index)
    else:
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check the camera connection and camera index."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return camera


def _capture_frames(camera, frame_count=BURST_FRAMES):
    frames = []
    for _ in range(frame_count):
        ok, frame = camera.read()
        if ok and frame is not None:
            frames.append(frame)
        time.sleep(BURST_INTERVAL)
    return frames


def _stability_score(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    gray_a = cv2.resize(gray_a, (320, 180))
    gray_b = cv2.resize(gray_b, (320, 180))

    return float(np.mean(cv2.absdiff(gray_a, gray_b)))


def _score_frame(frame, previous_frame=None):
    metrics = _quality_metrics(frame)

    blur = min(metrics["blur_score"], 1000.0)
    contrast = min(metrics["contrast"], 100.0)
    edge = min(metrics["center_edge_density"] * 1000.0, 100.0)

    stability_penalty = 0.0
    if previous_frame is not None:
        stability_penalty = min(_stability_score(frame, previous_frame), 50.0)

    score = (
        min(blur / 10.0, 100.0) * 0.55
        + contrast * 0.20
        + edge * 0.25
        - stability_penalty * 0.10
    )

    return score, metrics


def _crop_to_artwork(frame):
    """
    Conservative local crop to focus on the artwork.
    If no strong contour is found, use a gentle center crop.
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.04 * w * h:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.15 * w or bh < 0.15 * h:
            continue

        aspect = bw / max(1, bh)
        if aspect < 0.25 or aspect > 4.0:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_penalty = (
            abs(cx - w / 2.0) / max(1, w)
            + abs(cy - h / 2.0) / max(1, h)
        )

        score = area - center_penalty * 0.20 * w * h
        if score > best_score:
            best = (x, y, bw, bh)
            best_score = score

    if best is None:
        margin_x = int(w * 0.07)
        margin_y = int(h * 0.07)
        cropped = frame[margin_y : h - margin_y, margin_x : w - margin_x].copy()
        return cropped, None

    x, y, bw, bh = best
    pad = int(max(bw, bh) * 0.12)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    cropped = frame[y1:y2, x1:x2].copy()
    return cropped, (x1, y1, x2, y2)


def _draw_overlay(frame, lines):
    canvas = frame.copy()
    y = 34
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 32
    return canvas


def capture_artwork(
    camera_index=CAMERA_INDEX,
    preview_seconds=4.0,
    show_preview=True,
    auto_retry=True,
    retry_pause=0.9,
):
    """
    Capture and locally validate artwork from the webcam.

    This function NEVER calls Gemini.

    If show_preview=True, a live preview window is opened so the user can
    frame the artwork before the best frame is selected.

    If capture fails locally, it retries automatically until it gets a
    usable frame or the user cancels with Q / Esc.
    """
    camera = None
    attempt = 0

    try:
        camera = _open_camera(camera_index)
        time.sleep(WARMUP_SECONDS)

        while True:
            attempt += 1
            preview_frames = []
            preview_start = time.monotonic()
            window_name = "Art Expert Camera Preview"

            if show_preview:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            last_frame = None

            while True:
                ok, frame = camera.read()
                if ok and frame is not None:
                    last_frame = frame
                    preview_frames.append(frame)

                    metrics = _quality_metrics(frame)
                    quality = _quality_result(metrics)
                    elapsed = time.monotonic() - preview_start
                    remaining = max(0.0, preview_seconds - elapsed)

                    if show_preview:
                        status_line = (
                            "GOOD FRAME"
                            if quality["ok"]
                            else f"ISSUE: {quality['reason']}"
                        )

                        overlay = _draw_overlay(
                            frame,
                            [
                                f"Attempt {attempt}",
                                "Hold the artwork in the frame",
                                f"Auto-capture in {remaining:0.1f}s",
                                status_line,
                                "Press Enter to capture now",
                                "Press Q / Esc to cancel",
                            ],
                        )

                        cv2.imshow(window_name, overlay)
                        key = cv2.waitKey(1) & 0xFF

                        if key in (27, ord("q")):
                            cv2.destroyWindow(window_name)
                            return {
                                "success": False,
                                "retry": True,
                                "image_path": None,
                                "reason": "Capture cancelled by user.",
                                "metrics": {},
                            }

                        if key in (13, 32):
                            break

                    if elapsed >= preview_seconds:
                        break

                else:
                    time.sleep(0.02)

            if not preview_frames:
                reason = "The webcam did not return a usable frame."
                if not auto_retry:
                    return {
                        "success": False,
                        "retry": True,
                        "image_path": None,
                        "reason": reason,
                        "metrics": {},
                    }

                if show_preview and last_frame is not None:
                    retry_frame = _draw_overlay(
                        last_frame,
                        [
                            "CAPTURE FAILED",
                            reason,
                            "Retrying automatically...",
                            "Press Q / Esc to cancel",
                        ],
                    )
                    cv2.imshow(window_name, retry_frame)
                    cv2.waitKey(int(retry_pause * 1000))
                else:
                    time.sleep(retry_pause)

                if show_preview:
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                continue

            candidates = []
            previous = None

            for frame in preview_frames:
                score, metrics = _score_frame(frame, previous)
                candidates.append(
                    {
                        "frame": frame,
                        "score": score,
                        "quality": _quality_result(metrics),
                    }
                )
                previous = frame

            valid = [item for item in candidates if item["quality"]["ok"]]

            if not valid:
                best = max(candidates, key=lambda item: item["score"])
                reason = best["quality"]["reason"]

                if not auto_retry:
                    return {
                        "success": False,
                        "retry": True,
                        "image_path": None,
                        "reason": reason,
                        "metrics": best["quality"]["metrics"],
                    }

                if show_preview and last_frame is not None:
                    retry_frame = _draw_overlay(
                        last_frame,
                        [
                            "CAPTURE REJECTED",
                            reason,
                            "Retrying automatically...",
                            "Press Q / Esc to cancel",
                        ],
                    )
                    cv2.imshow(window_name, retry_frame)
                    cv2.waitKey(int(retry_pause * 1000))
                else:
                    time.sleep(retry_pause)

                if show_preview:
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                continue

            best = max(valid, key=lambda item: item["score"])

            cropped, crop_box = _crop_to_artwork(best["frame"])
            cropped_quality = _quality_result(_quality_metrics(cropped))

            if not cropped_quality["ok"]:
                reason = cropped_quality["reason"]

                if not auto_retry:
                    return {
                        "success": False,
                        "retry": True,
                        "image_path": None,
                        "reason": reason,
                        "metrics": cropped_quality["metrics"],
                    }

                if show_preview and last_frame is not None:
                    retry_frame = _draw_overlay(
                        last_frame,
                        [
                            "CAPTURE REJECTED AFTER CROP",
                            reason,
                            "Retrying automatically...",
                            "Press Q / Esc to cancel",
                        ],
                    )
                    cv2.imshow(window_name, retry_frame)
                    cv2.waitKey(int(retry_pause * 1000))
                else:
                    time.sleep(retry_pause)

                if show_preview:
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                continue

            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

            saved = cv2.imwrite(
                str(CAPTURE_PATH),
                cropped,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            if not saved:
                reason = "I captured the artwork, but could not save the image."

                if not auto_retry:
                    return {
                        "success": False,
                        "retry": True,
                        "image_path": None,
                        "reason": reason,
                        "metrics": cropped_quality["metrics"],
                    }

                if show_preview and last_frame is not None:
                    retry_frame = _draw_overlay(
                        last_frame,
                        [
                            "SAVE FAILED",
                            reason,
                            "Retrying automatically...",
                            "Press Q / Esc to cancel",
                        ],
                    )
                    cv2.imshow(window_name, retry_frame)
                    cv2.waitKey(int(retry_pause * 1000))
                else:
                    time.sleep(retry_pause)

                if show_preview:
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                continue

            if show_preview and crop_box is not None:
                x1, y1, x2, y2 = crop_box
                preview = best["frame"].copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.imshow(
                    window_name,
                    _draw_overlay(
                        preview,
                        [
                            "GOOD FRAME SELECTED",
                            "Cropping to the artwork and saving...",
                            "Press Q / Esc to cancel next time",
                        ],
                    ),
                )
                cv2.waitKey(250)

            return {
                "success": True,
                "retry": False,
                "image_path": CAPTURE_PATH,
                "reason": "Image captured successfully.",
                "metrics": cropped_quality["metrics"],
            }

    except Exception as exc:
        return {
            "success": False,
            "retry": True,
            "image_path": None,
            "reason": f"Camera error: {exc}",
            "metrics": {},
        }

    finally:
        if camera is not None:
            camera.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================
# ORIGINAL GEMINI IDENTIFICATION
# ============================================================

def identify_art(image_path: str | Path) -> dict:
    """
    Identifies the artwork from an image.

    Returns the original standardized Art Context dictionary that is
    passed to expert.py.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = read_text(BASE_DIR / "prompt.txt")
    image = load_image(image_path)

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image],
    )

    result = extract_json(response.text or "")

    art_context = {
        "art_style": result.get("art_style", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "origin": result.get("origin", "Unknown"),
        "key_features": result.get("key_features", []),
        "why": result.get("why", []),
        "needs_review": result.get("needs_review", True),
    }

    return art_context


def identify_art_from_camera(
    camera_index=CAMERA_INDEX,
    preview_seconds=4.0,
    show_preview=True,
    auto_retry=True,
    retry_pause=0.9,
):
    """
    Convenience helper:
        capture locally -> send accepted frame to Gemini.
    """
    capture = capture_artwork(
        camera_index=camera_index,
        preview_seconds=preview_seconds,
        show_preview=show_preview,
        auto_retry=auto_retry,
        retry_pause=retry_pause,
    )

    if not capture["success"]:
        return capture

    art_context = identify_art(capture["image_path"])
    return {
        "success": True,
        "retry": False,
        "image_path": capture["image_path"],
        "metrics": capture["metrics"],
        "art_context": art_context,
        "reason": capture["reason"],
    }


# ============================================================
# STANDALONE END-TO-END TEST
# ============================================================

def _print_quality(metrics):
    if not metrics:
        return

    print(f"Brightness : {metrics.get('brightness', 0):.1f}")
    print(f"Contrast   : {metrics.get('contrast', 0):.1f}")
    print(f"Blur score : {metrics.get('blur_score', 0):.1f}")
    print(f"Center edges: {metrics.get('center_edge_density', 0):.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ART EXPERT - CAMERA + GEMINI TEST")
    print("=" * 60)
    print()
    print("Opening webcam...")
    print("Hold the artwork inside the preview.")
    print("Local image checks run before Gemini.")
    print()

    capture = capture_artwork(
        preview_seconds=4.0,
        show_preview=True,
        auto_retry=True,
    )

    print()

    if not capture["success"]:
        print("CAPTURE FAILED")
        print(f"Reason: {capture['reason']}")
        print()
        print("Gemini was NOT called.")
        raise SystemExit(1)

    print("IMAGE CAPTURED")
    print(f"Saved to: {capture['image_path']}")
    print(f"Reason: {capture['reason']}")
    print()

    _print_quality(capture["metrics"])

    print()
    print("Sending the accepted image to Gemini...")
    print()

    try:
        result = identify_art(capture["image_path"])
        print("GEMINI IDENTIFICATION SUCCESS")
        print()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print("GEMINI IDENTIFICATION FAILED")
        print(f"Error: {exc}")
        raise