from gtts import gTTS
from playsound import playsound
import os
import uuid

def speak(text: str):
    sentences = text.split(". ")

    for sentence in sentences:
        if not sentence.strip():
            continue

        filename = f"voice_{uuid.uuid4().hex}.mp3"

        tts = gTTS(text=sentence, lang="en")
        tts.save(filename)

        playsound(filename)
        os.remove(filename)

