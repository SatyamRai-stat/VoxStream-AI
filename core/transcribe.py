from faster_whisper import WhisperModel
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Faster-Whisper model: {WHISPER_MODEL}...")

        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print("Model loaded.")

    return _model


def transcribe_chunk(chunk_path: str) -> str:
    model = load_model()

    segments, info = model.transcribe(
        chunk_path,
        beam_size=1
    )

    text = " ".join(segment.text for segment in segments)

    return text.strip()


def transcribe_all(chunks: list[str]) -> str:
    transcripts = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Transcribing chunk {i}/{len(chunks)}")

        text = transcribe_chunk(chunk)
        transcripts.append(text)

    return "\n".join(transcripts)
