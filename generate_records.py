import os
import sounddevice as sd
from scipy.io.wavfile import write

def grabar_audio(nombre_archivo, version, duracion=3, samplerate=44100):
    carpeta = os.path.join("audios","grabados", f"v{version}")
    os.makedirs(carpeta, exist_ok=True)

    print(f"Grabando '{nombre_archivo}.wav' en carpeta '{carpeta}'...")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    path = os.path.join(carpeta, f"{nombre_archivo}.wav")
    write(path, samplerate, audio)
    print(f"Guardado en: {path}")




version = 7
duracion = 60 # segundos
#grabar_audio("laberinto", version, duracion)
#grabar_audio("tedoyunacancion", version, duracion)
grabar_audio("besofuego", version, duracion)

