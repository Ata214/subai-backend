"""
SubAI Backend — Video Altyazı Uygulaması
=========================================
Gerçek zamanlı durum takibi (Polling) ve Groq Whisper entegrasyonu.
"""

import os
import json
import base64
import uuid
import shutil
import logging
import tempfile
import subprocess
import traceback
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ---------------------------------------------------------------------------
# Loglama ve Global Durum
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("subai")

# İşlem durumlarını tutacağımız sözlük (Bellekte tutulur)
# Yapısı: { "job_id": {"status": "processing|completed|error", "step": 0, "message": "", "video_path": "", "srt_content": "", "transcript_text": ""} }
JOBS = {}

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}

ASS_HEADER = """[Script Info]
Title: SubAI Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,85,&H00FFFFFF,&H00FFD400,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,2,30,30,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg bulunamadı! Lütfen ffmpeg yükleyin.")
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY ortam değişkeni ayarlanmamış!")
    yield

app = FastAPI(title="SubAI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-SRT-Base64", "X-Transcript-Text"],
)

# ===========================================================================
# Yardımcı Fonksiyonlar
# ===========================================================================

def _update_job(job_id: str, step: int, message: str, status: str = "processing", error: str = None):
    """İşlem durumunu günceller."""
    if job_id in JOBS:
        JOBS[job_id]["step"] = step
        JOBS[job_id]["message"] = message
        JOBS[job_id]["status"] = status
        if error:
            JOBS[job_id]["error"] = error
        logger.info(f"[{job_id}] Adım {step}: {message}")

def _extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Ses çıkarma başarısız: {result.stderr[:200]}")

def _transcribe_audio(audio_path: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY ortam değişkeni ayarlanmamış.")

    client = Groq(api_key=api_key)
    audio_size = os.path.getsize(audio_path)

    if audio_size > 25 * 1024 * 1024:
        compressed_path = audio_path.replace(".wav", "_compressed.mp3")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path, "-acodec", "libmp3lame",
            "-b:a", "64k", "-ar", "16000", "-ac", "1", compressed_path,
        ]
        if subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0:
            audio_path = compressed_path

    # Kullanıcının belirttiği kod parçası eklendi: file.read() ve temperature=0
    filename = Path(audio_path).name
    with open(audio_path, "rb") as file:
        file_bytes = file.read()
        
    transcription = client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3",
        temperature=0,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )
    return transcription

def _format_srt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"

def _format_ass_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int((seconds % 1) * 100)
    return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}.{centis:02d}"

def _generate_srt(segments: list, srt_path: str) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(f"{i}\n{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}\n{seg['text'].strip()}\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _generate_ass_karaoke(segments: list, words: list, ass_path: str) -> None:
    dialogue_lines = []
    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text: continue
        
        seg_words = []
        if words:
            for w in words:
                if w["start"] >= seg["start"] - 0.05 and w["start"] < seg["end"] + 0.05:
                    seg_words.append(w)
                    
        if seg_words:
            parts = []
            for i, w in enumerate(seg_words):
                dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                parts.append(f"{{\\kf{dur_cs}}}{w['word']}")
                if i < len(seg_words) - 1:
                    gap = seg_words[i + 1]["start"] - w["end"]
                    parts.append(f"{{\\kf{max(1, int(gap * 100))}}} " if gap > 0.01 else " ")
            karaoke_text = "".join(parts)
        else:
            karaoke_text = f"{{\\kf{int((seg['end'] - seg['start']) * 100)}}}{seg_text}"

        start_time, end_time = _format_ass_time(seg["start"]), _format_ass_time(seg["end"])
        dialogue_lines.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}")

    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ASS_HEADER + "\n".join(dialogue_lines) + "\n")

def _hardcode_subtitles(video_path: str, ass_path: str, output_path: str) -> None:
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vf", f"ass='{escaped_ass}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Altyazı gömme başarısız: {result.stderr[:200]}")

# ===========================================================================
# Background Task: Video İşleme
# ===========================================================================

def _translate_segments(segments: list, target_language: str) -> list:
    """Groq Llama3 ile segment metinlerini çevirir ve karaoke için pseudo-kelimeler üretir."""
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    texts = [seg["text"] for seg in segments]
    
    prompt = f"Translate the following JSON array of strings into {target_language}. Return ONLY a valid JSON array of strings in the exact same order. Do not add any markdown formatting, code blocks, or explanations. Just the raw JSON array.\n\nJSON: {json.dumps(texts)}"
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional translator. You only output valid JSON arrays."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
        temperature=0,
    )
    
    try:
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        translated_texts = json.loads(content)
        
        new_words = []
        for i, seg in enumerate(segments):
            if i < len(translated_texts):
                seg["text"] = translated_texts[i]
                t_words = translated_texts[i].split()
                total_duration = seg["end"] - seg["start"]
                total_chars = sum(len(w) for w in t_words)
                
                current_time = seg["start"]
                for w in t_words:
                    word_duration = total_duration * (len(w) / max(1, total_chars))
                    new_words.append({
                        "word": w,
                        "start": current_time,
                        "end": current_time + word_duration
                    })
                    current_time += word_duration
        return new_words
    except Exception as e:
        logger.error(f"Çeviri hatası: {e}")
        return None

def process_video_task(job_id: str, temp_dir: str, video_input: str, ext: str, original_filename: str, language: str = "auto"):
    """Arka planda videoyu işleyen ana fonksiyon."""
    try:
        audio_path = os.path.join(temp_dir, "audio.wav")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        ass_path = os.path.join(temp_dir, "subtitles.ass")
        video_output = os.path.join(temp_dir, "output.mp4")

        # 1. Ses Çıkarma
        _update_job(job_id, 1, "Sesi videodan ayırıyorum...")
        _extract_audio(video_input, audio_path)

        # 2. Transkripsiyon
        _update_job(job_id, 2, "AI sesleri metne döküyor...")
        transcription = _transcribe_audio(audio_path)

        def safe_get(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        raw_segments = safe_get(transcription, 'segments', [])
        
        # 150ms erken başlatma (offset)
        offset = 0.15
        
        segments = [{"start": max(0, safe_get(s, 'start', 0) - offset), "end": max(0, safe_get(s, 'end', 0) - offset), "text": safe_get(s, 'text', '')} 
                   for s in raw_segments]

        raw_words = safe_get(transcription, 'words', [])
        words = [{"word": safe_get(w, 'word', ''), "start": max(0, safe_get(w, 'start', 0) - offset), "end": max(0, safe_get(w, 'end', 0) - offset)} 
                for w in raw_words]

        if not segments:
            raise ValueError("Videoda konuşma algılanamadı.")

        # Çeviri Gerekli mi?
        detected_lang = safe_get(transcription, 'language', 'auto')
        logger.info(f"[{job_id}] Algılanan dil: {detected_lang}, İstenen dil: {language}")
        
        # Auto değilse ve diller uyuşmuyorsa çeviri yap
        if language != "auto" and not detected_lang.startswith(language.split('-')[0]):
            _update_job(job_id, 3, "Altyazılar çevriliyor...")
            new_words = _translate_segments(segments, language)
            if new_words:
                words = new_words

        # 3. Altyazı Oluşturma
        _update_job(job_id, 4, "Karaoke altyazıları hesaplıyorum...")
        _generate_srt(segments, srt_path)
        _generate_ass_karaoke(segments, words, ass_path)

        # 4. Videoya Gömme
        _update_job(job_id, 5, "Altyazılar videoya gömülüyor (Bu biraz sürebilir)...")
        _hardcode_subtitles(video_input, ass_path, video_output)

        # 5. Tamamlama
        srt_content = ""
        if os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_content = f.read()

        transcript_text = safe_get(transcription, 'text', '')

        JOBS[job_id].update({
            "status": "completed",
            "step": 6,
            "message": "İşlem tamamlandı! 🎉",
            "video_output": video_output,
            "srt_content": srt_content,
            "transcript_text": transcript_text,
            "original_filename": original_filename,
            "temp_dir": temp_dir # Cleanup için kaydediyoruz
        })
        logger.info(f"[{job_id}] İşlem tüm adımlarıyla tamamlandı.")

    except Exception as e:
        logger.error(f"[{job_id}] Hata: {traceback.format_exc()}")
        _update_job(job_id, 0, "Hata oluştu", status="error", error=str(e))


# ===========================================================================
# API Endpoint'leri
# ===========================================================================

@app.get("/api/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.post("/api/transcribe")
async def start_transcription(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Videoyu yükler ve işlemi arka planda başlatır."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı belirtilmedi.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı.")

    job_id = uuid.uuid4().hex[:8]
    temp_dir = tempfile.mkdtemp(prefix="subai_")
    video_input = os.path.join(temp_dir, f"input{ext}")

    # JOBS sözlüğüne ekle
    JOBS[job_id] = {
        "status": "processing",
        "step": 0,
        "message": "Video yükleniyor ve kaydediliyor...",
        "error": None
    }

    try:
        # Videoyu diske kaydet
        with open(video_input, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                
        # Arka plan görevini başlat
        background_tasks.add_task(process_video_task, job_id, temp_dir, video_input, ext, file.filename, language)
        
        return JSONResponse(content={"job_id": job_id, "message": "İşlem başlatıldı."})

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """İşlem durumunu döndürür."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job bulunamadı.")
    
    job = JOBS[job_id]
    return JSONResponse(content={
        "status": job["status"],
        "step": job["step"],
        "message": job["message"],
        "error": job.get("error")
    })

@app.get("/api/download/{job_id}")
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    """Tamamlanan videoyu indirir ve geçici dosyaları siler."""
    if job_id not in JOBS or JOBS[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video henüz hazır değil.")

    job = JOBS[job_id]
    
    srt_b64 = base64.b64encode(job["srt_content"].encode("utf-8")).decode("ascii")
    transcript_b64 = base64.b64encode(job["transcript_text"][:1000].encode("utf-8")).decode("ascii")

    # İndirme bittikten sonra temizlik yapacak fonksiyon
    def cleanup():
        try:
            shutil.rmtree(job["temp_dir"], ignore_errors=True)
            del JOBS[job_id]
            logger.info(f"[{job_id}] Kaynaklar temizlendi.")
        except Exception:
            pass

    response = FileResponse(
        path=job["video_output"],
        media_type="video/mp4",
        filename=f"subai_{Path(job['original_filename']).stem}.mp4",
        headers={
            "X-SRT-Base64": srt_b64,
            "X-Transcript-Text": transcript_b64,
            "Access-Control-Expose-Headers": "X-SRT-Base64, X-Transcript-Text",
        },
        background=BackgroundTasks()
    )
    response.background.add_task(cleanup)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
