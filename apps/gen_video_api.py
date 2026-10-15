"""Local, torch-free VieNeu v3 Turbo adapter for gen-video."""
import base64
import binascii
import io
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from vieneu import Vieneu

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vieneu.gen-video")
model = None
gate = threading.BoundedSemaphore(1)


@asynccontextmanager
async def lifespan(app):
    global model
    log.info("Loading local VieNeu v3 Turbo ONNX/CPU fp32")
    model = Vieneu(mode="v3turbo", backend="onnx", device="cpu", precision="fp32",
                   threads=int(os.environ.get("VIENEU_THREADS", "6")))
    # Load lazy cloning graphs before health becomes ready. Both the preset and
    # clone paths must be cached so offline restarts don't need Hugging Face.
    with tempfile.TemporaryDirectory(prefix="vieneu-warmup-") as folder:
        reference = Path(folder) / "reference.wav"
        audio = model.infer("Xin chào, đây là mẫu giọng để khởi động hệ thống.", apply_watermark=False)
        sf.write(reference, audio, model.sample_rate, subtype="PCM_16")
        model.infer("Xin chào.", ref_audio=str(reference), denoise=True, apply_watermark=False)
    log.info("Local preset + cloning engine ready, SDK 3.8.3, %s Hz", model.sample_rate)
    yield
    model = None


app = FastAPI(title="VieNeu local · gen-video", version="3.8.3", lifespan=lifespan)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = None
    reference_audio: str | None = Field(default=None, max_length=10_000_000)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5, allow_inf_nan=False)
    max_new_frames: int = Field(default=300, ge=50, le=1200)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(503, "Model loading")
    return {"status": "ok", "backend": "onnx", "device": "cpu", "precision": "fp32", "sample_rate": model.sample_rate,
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1"}


@app.get("/voices")
def voices():
    if model is None:
        raise HTTPException(503, "Model loading")
    items = []
    for name, metadata in model._preset_voices.items():
        gender = str(metadata.get("gender", "")).lower()
        items.append({"id": name, "name": name, "gender": "female" if gender in ("f", "female", "nữ", "nu") else "male",
                      "style": metadata.get("description", ""), "aliases": metadata.get("aliases", [])})
    return {"voices": items}


def decode_reference(value: str, folder: Path) -> str:
    match = re.fullmatch(r"data:(audio/[a-zA-Z0-9.+-]+)(?:;codecs=[^;,]+)?;base64,([a-zA-Z0-9+/=\s]+)", value)
    if not match:
        raise HTTPException(400, "Audio mẫu không hợp lệ")
    try:
        audio = base64.b64decode(re.sub(r"\s", "", match[2]), validate=True)
    except binascii.Error:
        raise HTTPException(400, "Base64 audio không hợp lệ")
    if not audio or len(audio) > 7_000_000:
        raise HTTPException(400, "Audio mẫu phải nhỏ hơn 7 MB")
    raw = folder / "input.audio"
    wav = folder / "reference.wav"
    raw.write_bytes(audio)
    # Browser recordings may be WebM/Opus or M4A. Decode by file contents,
    # convert to mono PCM, cap the reference at the SDK's 8-second limit.
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                        "-t", "8", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav)],
                       check=True, capture_output=True, timeout=30)
        info = sf.info(wav)
        if info.duration < 0.5:
            raise HTTPException(400, "Audio mẫu quá ngắn")
    except (subprocess.SubprocessError, RuntimeError) as error:
        raise HTTPException(400, "Không giải mã được audio mẫu. Hãy dùng WAV, MP3, M4A hoặc WebM.") from error
    return str(wav)


@app.post("/synthesize", response_class=Response)
def synthesize(request: SpeechRequest):
    if model is None:
        raise HTTPException(503, "Model loading")
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Văn bản không được để trống")
    if not gate.acquire(timeout=60):
        raise HTTPException(429, "VieNeu local đang bận. Vui lòng thử lại.")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="vieneu-request-") as directory:
            kwargs = {"temperature": request.temperature, "top_k": 25, "top_p": 0.95,
                      "repetition_penalty": 1.2, "max_new_frames": request.max_new_frames,
                      "max_chars": 256, "apply_watermark": False}
            if request.reference_audio:
                kwargs["ref_audio"] = decode_reference(request.reference_audio, Path(directory))
                kwargs["denoise"] = True
            elif request.voice:
                resolved = model.resolve_voice_name(request.voice)
                if resolved is None:
                    raise HTTPException(400, f"Giọng '{request.voice}' không có trong model local. Hãy tải lại danh sách giọng.")
                kwargs["voice"] = resolved
            waveform = np.asarray(model.infer(text, **kwargs), dtype=np.float32)
            output = io.BytesIO()
            sf.write(output, waveform, model.sample_rate, format="WAV", subtype="PCM_16")
            log.info("Synthesis completed: cloned=%s chars=%d audio=%.2fs elapsed=%.2fs",
                     bool(request.reference_audio), len(text), len(waveform) / model.sample_rate, time.monotonic() - started)
            return Response(output.getvalue(), media_type="audio/wav", headers={"X-VieNeu-Source": "local", "Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as error:
        log.exception("Local synthesis failed")
        raise HTTPException(500, "VieNeu local xử lý thất bại; kiểm tra log container.") from error
    finally:
        gate.release()
