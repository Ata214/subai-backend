"""
SubAI Backend — Video Altyazı Uygulaması
=========================================
Bu sunucu, video dosyalarını alır, ses çıkarır, Groq Whisper API ile
transkript oluşturur ve karaoke tarzı (TikTok/Reels stili) altyazılı
video üretir.

Endpoints:
  POST /api/transcribe  — Video yükle, altyazılı video + SRT dosyası al
  GET  /api/health      — Sunucu sağlık kontrolü
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
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ---------------------------------------------------------------------------
# Loglama ayarları
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("subai")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}

# ASS altyazı stili — TikTok/Reels tarzı karaoke efekti
# PrimaryColour: Beyaz (&H00FFFFFF), SecondaryColour: Neon Cyan (&H00FFD400 — BGR formatı)
# OutlineColour: Siyah, Outline: 2px, Shadow: 1px, Bold: 1
ASS_HEADER = """[Script Info]
Title: SubAI Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFD400,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,30,30,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ---------------------------------------------------------------------------
# Uygulama yaşam döngüsü — başlangıçta kontroller, kapanışta temizlik
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken gerekli kontrolleri yap."""
    # ffmpeg kurulu mu kontrol et
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg bulunamadı! Lütfen ffmpeg yükleyin.")
        raise RuntimeError("ffmpeg is required but not found on PATH")

    # Groq API anahtarı var mı kontrol et
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY ortam değişkeni ayarlanmamış!")

    logger.info("SubAI backend başlatıldı ✓")
    yield
    logger.info("SubAI backend kapatılıyor...")


# ---------------------------------------------------------------------------
# FastAPI uygulaması
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SubAI Backend",
    description="Video altyazı oluşturma API'si — Groq Whisper + Karaoke efekti",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS ayarları — Flutter uygulamasından gelen isteklere izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Yardımcı Fonksiyonlar
# ===========================================================================


def _validate_extension(filename: str) -> str:
    """Dosya uzantısını kontrol et ve döndür."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {ext}. "
                   f"Desteklenen formatlar: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


def _extract_audio(video_path: str, audio_path: str) -> None:
    """
    ffmpeg ile videodan ses çıkar.
    Whisper için 16kHz mono WAV formatında çıkarılır.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # Video akışını devre dışı bırak
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz örnekleme hızı
        "-ac", "1",               # Mono kanal
        audio_path,
    ]
    logger.info(f"Ses çıkarılıyor: {Path(video_path).name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"ffmpeg ses çıkarma hatası: {result.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Ses çıkarma başarısız: {result.stderr[:500]}",
        )
    logger.info("Ses başarıyla çıkarıldı ✓")


def _transcribe_audio(audio_path: str) -> dict:
    """
    Groq Whisper API ile ses dosyasını transkript et.
    verbose_json formatında, kelime düzeyinde zaman damgaları ile.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY ortam değişkeni ayarlanmamış.",
        )

    client = Groq(api_key=api_key)

    # Ses dosyası boyutunu kontrol et (Groq limiti: 25MB)
    audio_size = os.path.getsize(audio_path)
    logger.info(f"Ses dosyası boyutu: {audio_size / (1024*1024):.1f} MB")

    if audio_size > 25 * 1024 * 1024:
        # Ses dosyası çok büyükse, sıkıştırılmış formata çevir
        compressed_path = audio_path.replace(".wav", "_compressed.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-acodec", "libmp3lame",
            "-b:a", "64k",        # Düşük bitrate — küçük dosya boyutu
            "-ar", "16000",
            "-ac", "1",
            compressed_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Ses sıkıştırma başarısız.")
        audio_path = compressed_path
        logger.info(f"Sıkıştırılmış boyut: {os.path.getsize(audio_path) / (1024*1024):.1f} MB")

    logger.info("Groq Whisper API'ye gönderiliyor...")
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(Path(audio_path).name, audio_file),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except Exception as e:
        logger.error(f"Groq API hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Groq API hatası: {str(e)[:300]}",
        )

    logger.info(f"Transkripsiyon tamamlandı ✓")
    return transcription


def _format_srt_time(seconds: float) -> str:
    """Saniyeyi SRT zaman formatına çevir: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_ass_time(seconds: float) -> str:
    """Saniyeyi ASS zaman formatına çevir: H:MM:SS.cc (centisaniye)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _generate_srt(segments: list, srt_path: str) -> None:
    """Segment listesinden standart SRT altyazı dosyası oluştur."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"SRT dosyası oluşturuldu ✓")


def _clean_word(word: str) -> str:
    """Kelimeyi temizle — gereksiz boşlukları kaldır."""
    return word.strip()


def _generate_ass_karaoke(segments: list, words: list, ass_path: str) -> None:
    """
    TikTok/Reels tarzı karaoke efektli ASS altyazı dosyası oluştur.
    Kelime bilgisi yoksa, segment bazlı basit altyazı oluşturulur.
    """
    dialogue_lines = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()

        if not seg_text:
            continue

        if words:
            # Bu segmente ait kelimeleri bul
            seg_words = _find_words_for_segment(words, seg_start, seg_end, seg_text)

            if seg_words:
                # Kelime bazlı karaoke efekti oluştur
                karaoke_text = _build_karaoke_line(seg_words, seg_start)
            else:
                # Kelime bulunamazsa basit metin
                duration_cs = int((seg_end - seg_start) * 100)
                karaoke_text = f"{{\\kf{duration_cs}}}{seg_text}"
        else:
            # Kelime bilgisi hiç yoksa basit metin
            duration_cs = int((seg_end - seg_start) * 100)
            karaoke_text = f"{{\\kf{duration_cs}}}{seg_text}"

        start_time = _format_ass_time(seg_start)
        end_time = _format_ass_time(seg_end)

        dialogue_lines.append(
            f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}"
        )

    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(dialogue_lines))
        f.write("\n")

    logger.info(f"ASS karaoke dosyası oluşturuldu ✓")


def _find_words_for_segment(
    words: list, seg_start: float, seg_end: float, seg_text: str
) -> list:
    """Verilen segmente ait kelimeleri zaman damgalarına göre bul."""
    matched = []
    tolerance = 0.05  # 50ms tolerans

    for w in words:
        w_start = w.get("start", 0)
        w_end = w.get("end", 0)
        w_text = w.get("word", "").strip()

        if not w_text:
            continue

        if (w_start >= seg_start - tolerance) and (w_start < seg_end + tolerance):
            matched.append({
                "word": w_text,
                "start": w_start,
                "end": w_end,
            })

    return matched


def _build_karaoke_line(seg_words: list, seg_start: float) -> str:
    """Kelime listesinden ASS karaoke satırı oluştur."""
    parts = []

    for i, w in enumerate(seg_words):
        word_text = w["word"]

        # Kelime süresini centisaniye olarak hesapla
        duration_sec = w["end"] - w["start"]
        duration_cs = max(1, int(duration_sec * 100))

        parts.append(f"{{\\kf{duration_cs}}}{word_text}")

        if i < len(seg_words) - 1:
            gap = seg_words[i + 1]["start"] - w["end"]
            if gap > 0.01:
                gap_cs = max(1, int(gap * 100))
                parts.append(f"{{\\kf{gap_cs}}} ")
            else:
                parts.append(" ")

    return "".join(parts)


def _hardcode_subtitles(
    video_path: str, ass_path: str, output_path: str
) -> None:
    """ffmpeg ile ASS altyazıları videoya gömülü olarak yaz (hardcode)."""
    # ASS dosya yolundaki ters slash ve iki noktayı escape et
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass='{escaped_ass}'",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info("Altyazılar videoya gömülüyor...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"ffmpeg altyazı gömme hatası: {result.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Altyazı gömme başarısız: {result.stderr[:500]}",
        )
    logger.info("Video altyazıları başarıyla gömüldü ✓")


# ===========================================================================
# API Endpoint'leri
# ===========================================================================


@app.get("/api/health")
async def health_check():
    """Sunucu sağlık kontrolü endpoint'i."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))

    return JSONResponse(
        content={
            "status": "healthy" if (ffmpeg_ok and groq_key_set) else "degraded",
            "service": "SubAI Backend",
            "version": "1.0.0",
            "checks": {
                "ffmpeg": "available" if ffmpeg_ok else "missing",
                "groq_api_key": "configured" if groq_key_set else "missing",
            },
        }
    )


@app.post("/api/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    """
    Video dosyasını alır, transkript oluşturur ve JSON yanıt döndürür.
    Video dosyası ayrı endpoint üzerinden indirilebilir.
    """
    # --- Dosya doğrulama ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı belirtilmedi.")

    ext = _validate_extension(file.filename)

    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya çok büyük. Maksimum: {MAX_UPLOAD_SIZE // (1024*1024)} MB",
        )

    # --- Geçici dizin oluştur ---
    temp_dir = tempfile.mkdtemp(prefix="subai_")
    job_id = uuid.uuid4().hex[:8]

    try:
        logger.info(f"[{job_id}] İş başlatıldı — Dosya: {file.filename}")

        # Yollar
        video_input = os.path.join(temp_dir, f"input{ext}")
        audio_path = os.path.join(temp_dir, "audio.wav")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        ass_path = os.path.join(temp_dir, "subtitles.ass")
        video_output = os.path.join(temp_dir, "output.mp4")

        # 1. Video dosyasını diske kaydet
        logger.info(f"[{job_id}] Video diske kaydediliyor...")
        with open(video_input, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        file_size_mb = os.path.getsize(video_input) / (1024 * 1024)
        logger.info(f"[{job_id}] Video kaydedildi: {file_size_mb:.1f} MB")

        # 2. Ses çıkar
        logger.info(f"[{job_id}] Adım 2: Ses çıkarılıyor...")
        _extract_audio(video_input, audio_path)

        # 3. Groq Whisper ile transkript oluştur
        logger.info(f"[{job_id}] Adım 3: Transkripsiyon yapılıyor...")
        transcription = _transcribe_audio(audio_path)

        # Segment ve kelime verilerini çıkar (güvenli erişim)
        segments = []
        raw_segments = getattr(transcription, 'segments', None) or []
        for seg in raw_segments:
            try:
                segments.append({
                    "start": getattr(seg, 'start', 0) or 0,
                    "end": getattr(seg, 'end', 0) or 0,
                    "text": getattr(seg, 'text', '') or '',
                })
            except Exception as e:
                logger.warning(f"[{job_id}] Segment ayrıştırma hatası: {e}")
                continue

        words = []
        raw_words = getattr(transcription, 'words', None) or []
        for w in raw_words:
            try:
                words.append({
                    "word": getattr(w, 'word', '') or '',
                    "start": getattr(w, 'start', 0) or 0,
                    "end": getattr(w, 'end', 0) or 0,
                })
            except Exception as e:
                logger.warning(f"[{job_id}] Kelime ayrıştırma hatası: {e}")
                continue

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="Videoda konuşma algılanamadı. Lütfen ses içeren bir video yükleyin.",
            )

        logger.info(
            f"[{job_id}] Transkripsiyon: {len(segments)} segment, {len(words)} kelime"
        )

        # 4. SRT dosyası oluştur
        logger.info(f"[{job_id}] Adım 4: SRT oluşturuluyor...")
        _generate_srt(segments, srt_path)

        # 5. ASS karaoke dosyası oluştur
        logger.info(f"[{job_id}] Adım 5: ASS karaoke oluşturuluyor...")
        _generate_ass_karaoke(segments, words, ass_path)

        # 6. Altyazıları videoya göm
        logger.info(f"[{job_id}] Adım 6: Video oluşturuluyor...")
        _hardcode_subtitles(video_input, ass_path, video_output)

        # 7. Çıktı dosyasının varlığını doğrula
        if not os.path.exists(video_output):
            raise HTTPException(
                status_code=500,
                detail="Video oluşturma başarısız — çıktı dosyası bulunamadı.",
            )

        output_size_mb = os.path.getsize(video_output) / (1024 * 1024)
        logger.info(f"[{job_id}] Çıktı video boyutu: {output_size_mb:.1f} MB")

        # SRT içeriğini oku
        srt_content = ""
        if os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_content = f.read()

        # SRT'yi base64 ile encode et (header'da Türkçe karakter sorunu çözmek için)
        srt_b64 = base64.b64encode(srt_content.encode("utf-8")).decode("ascii")

        transcript_text = getattr(transcription, 'text', '') or ''

        logger.info(f"[{job_id}] İş tamamlandı ✓ — Video gönderiliyor...")

        # Video dosyasını döndür
        response = FileResponse(
            path=video_output,
            media_type="video/mp4",
            filename=f"subai_{Path(file.filename).stem}.mp4",
            headers={
                "X-SRT-Base64": srt_b64,
                "X-Transcript-Text": base64.b64encode(
                    transcript_text[:1000].encode("utf-8")
                ).decode("ascii"),
                "X-Segment-Count": str(len(segments)),
                "X-Word-Count": str(len(words)),
                "Access-Control-Expose-Headers": (
                    "X-SRT-Base64, X-Transcript-Text, X-Segment-Count, X-Word-Count"
                ),
            },
            background=_create_cleanup_task(temp_dir, job_id),
        )

        return response

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.exception(f"[{job_id}] Beklenmeyen hata: {e}")
        logger.error(f"[{job_id}] Traceback: {traceback.format_exc()}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"İşlem sırasında beklenmeyen bir hata oluştu: {str(e)[:300]}",
        )


def _create_cleanup_task(temp_dir: str, job_id: str):
    """FileResponse tamamlandıktan sonra geçici dizini temizle."""
    from starlette.background import BackgroundTask

    async def cleanup():
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"[{job_id}] Geçici dosyalar temizlendi ✓")
        except Exception as e:
            logger.warning(f"[{job_id}] Temizlik hatası: {e}")

    return BackgroundTask(cleanup)


# ===========================================================================
# Uygulama başlatma (doğrudan çalıştırma için)
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
