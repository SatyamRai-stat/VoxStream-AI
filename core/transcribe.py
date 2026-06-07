import whisper
import os
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 

def transcribe_chunk(chunk_path:str)->str:
    print(f"Transcribing: {chunk_path}")
    model=load_model()
    result=model.transcribe(chunk_path)
    print("Finished!")
    return result['text']

def transcribe_all(chunks:list):
    full_transcript=""
    for i, chunk in enumerate(chunks):
        text=transcribe_chunk(chunk)
        print(f"Transcribing chunk {i}/{len(chunks)}")
        full_transcript+=text+" "
    
    return full_transcript