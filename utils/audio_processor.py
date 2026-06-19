import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# Taking youtube url and getting out the audio file in {.wav} format
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,

        # ── Cloud server SSL + IP block fixes ────────────────────────────────
        "nocheckcertificate": True,
        "no_check_certificate": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        "extractor_args": {
            "youtube": {
                "skip": ["dash", "hls"],
                "player_skip": ["webpage", "configs", "js"],
            }
        },
        # Use IPv4 only — IPv6 often blocked on cloud servers
        "source_address": "0.0.0.0",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = (
            ydl.prepare_filename(info)
            .replace(".webm", ".wav")
            .replace(".m4a", ".wav")
            .replace(".mp4", ".wav")
        )
    return filename


# Any audio with different extensions(.mp3 etc) converting them into {.wav} format
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


# Converting large audio into chunks of 10 minutes
def chunks_audio(wav_path: str, chunks_min: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunks_mms = chunks_min * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunks_mms)):
        chunk = audio[start:start + chunks_mms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# All above functions into single function
def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        try:
            wav_path = download_youtube_audio(source)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download YouTube audio. "
                f"This may be due to YouTube blocking cloud server IPs. "
                f"Try uploading the audio file directly instead.\n"
                f"Original error: {str(e)}"
            )
    else:
        wav_path = convert_to_wav(source)

    chunks = chunks_audio(wav_path)
    return chunks