from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from io import BytesIO
import hashlib
import os
import asyncio

from TTS.api import TTS

app = FastAPI(title="TTS FastAPI Service", version="0.1")

MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/pt/cv/vits")
print("Loading TTS model:", MODEL_NAME)
try:
    tts = TTS(model_name=MODEL_NAME)
except Exception as e:
    print("Error loading TTS model:", e)
    tts = None


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    format: str | None = "mp3"  # mp3, wav


def text_hash(text: str, voice: str | None, fmt: str) -> str:
    h = hashlib.sha256((text + (voice or "") + fmt).encode("utf-8"))
    return h.hexdigest()


@app.post("/tts")
async def synthesize(req: TTSRequest):
    if not tts:
        raise HTTPException(status_code=500, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    fmt = (req.format or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="format must be 'mp3' or 'wav'")

    # length guard
    if len(text) > 2000:
        raise HTTPException(status_code=413, detail="text too long")

    cache_dir = os.getenv("TTS_CACHE_DIR", "/tmp/tts_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = text_hash(text, req.voice, fmt)
    out_path = os.path.join(cache_dir, f"{key}.{fmt}")

    if os.path.exists(out_path):
        def iterfile():
            with open(out_path, "rb") as f:
                yield from f

        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return StreamingResponse(iterfile(), media_type=media_type)

    # generate audio using TTS
    try:
        # Coqui TTS provides tts.tts_to_file or tts.tts (return wav numpy)
        # Use tts.tts_to_file for convenience
        # If voice selection is supported by model, pass language/speaker arguments as needed
        temp_out = out_path + ".part"
        # For mp3, generate wav then convert to mp3 using ffmpeg via pydub or system call.
        if fmt == "wav":
            tts.tts_to_file(text=text, file_path=temp_out, speaker=req.voice)
            os.rename(temp_out, out_path)
            def iterfile():
                with open(out_path, "rb") as f:
                    yield from f
            return StreamingResponse(iterfile(), media_type="audio/wav")
        else:
            # generate wav first
            wav_tmp = out_path + ".wav"
            tts.tts_to_file(text=text, file_path=wav_tmp, speaker=req.voice)
            # convert wav to mp3 using ffmpeg (requires ffmpeg installed)
            mp3_tmp = out_path + ".mp3"
            # using ffmpeg via system call
            cmd = f"ffmpeg -y -i {wav_tmp} -codec:a libmp3lame -qscale:a 2 {mp3_tmp}"
            rc = os.system(cmd)
            if rc != 0 or not os.path.exists(mp3_tmp):
                # fallback: return wav
                with open(wav_tmp, "rb") as f:
                    return StreamingResponse(f, media_type="audio/wav")
            os.rename(mp3_tmp, out_path)
            os.remove(wav_tmp)
            def iterfile():
                with open(out_path, "rb") as f:
                    yield from f
            return StreamingResponse(iterfile(), media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def voices():
    info = {"model": MODEL_NAME}
    try:
        speakers = getattr(tts, "speakers", None)
        languages = getattr(tts, "languages", None)
        if speakers:
            info["speakers"] = speakers
        if languages:
            info["languages"] = languages
    except Exception:
        pass
    return JSONResponse(info)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(tts)}
