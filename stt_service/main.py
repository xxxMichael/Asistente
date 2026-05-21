import pyaudio
import numpy as np
import requests
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
import wave
import time
import asyncio
import edge_tts
import pygame
import os
# 1. Configuración de Modelos
print("Cargando modelos... (Esto toma unos segundos la primera vez)")
# Carga el modelo de Wake Word (puedes usar "alexa", "hey_mycroft", "hey_jarvis")
# Carga tu modelo local personalizado
oww_model = Model(
    wakeword_models=["./models/goo_stahv.onnx"]
)  # Carga el modelo local de Whisper (modelo tiny para que sea rápido en la Mac 2012)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

pygame.mixer.init()
# Configuración de n8n
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/escucha"

# 2. Configuración del Micrófono (PyAudio)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)


def grabar_comando(duracion_segundos=5, archivo_salida="comando.wav"):
    """Graba el audio después de escuchar el Wake Word"""
    print("🎤 Escuchando comando...")
    frames = []
    # Grabación simple por tiempo (puedes mejorarlo luego para que detecte silencios)
    for _ in range(0, int(RATE / CHUNK * duracion_segundos)):
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    # Guardar en archivo temporal
    wf = wave.open(archivo_salida, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    return archivo_salida


def transcribir_y_enviar(archivo_audio):
    """Convierte el audio a texto y lo envía a n8n"""
    print("🧠 Transcribiendo...")
    segments, info = whisper_model.transcribe(
        archivo_audio, beam_size=5, language="es", condition_on_previous_text=False
    )
    texto = " ".join([segment.text for segment in segments]).strip()

    if texto:
        print(f"✅ Entendido: '{texto}'")
        # Enviar a n8n y ESPERAR respuesta
        try:
            print("⏳ Esperando respuesta de Gustav...")
            respuesta = requests.post(N8N_WEBHOOK_URL, json={"texto": texto})

            # Convertimos la respuesta de n8n a formato JSON
            datos = respuesta.json()

            # n8n suele devolver una lista. Extraemos el texto del campo 'output' de Gemini
            if isinstance(datos, list) and len(datos) > 0 and "output" in datos[0]:
                mensaje_ia = datos[0]["output"]
            else:
                # Por si n8n lo envía en otro formato
                mensaje_ia = respuesta.text

            print(f"\n🤖 Gustav dice: {mensaje_ia}\n")
            hablar(mensaje_ia)

        except Exception as e:
            print("❌ Error conectando con n8n:", e)
    else:
        print("🤷 No se entendió el comando.")


print("🟢 Sistema listo. Di 'Gustav' para empezar...")


def hablar(texto):
    """Convierte texto a voz usando Edge TTS (Voz masculina y rápida) y lo reproduce"""
    try:
        archivo_voz = "respuesta.mp3"
        
        # Configuración de voz y velocidad
        voz = "es-EC-LuisNeural" 
        velocidad = "+20%" # Prueba con +30% o +40% si lo quieres más rápido
        
        # Generamos el audio (edge-tts requiere ejecución asíncrona)
        async def generar_voz():
            comunicador = edge_tts.Communicate(texto, voz, rate=velocidad)
            await comunicador.save(archivo_voz)
            
        asyncio.run(generar_voz())
        
        # Reproduce el audio
        pygame.mixer.music.load(archivo_voz)
        pygame.mixer.music.play()
        
        # Espera a que termine de hablar antes de continuar
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        # --- AQUÍ ESTÁ LA CORRECCIÓN DE ESPACIOS ---
        # Descarga el archivo de la memoria y lo borra (afuera del while)
        pygame.mixer.music.unload()
        if os.path.exists(archivo_voz):
            os.remove(archivo_voz)
            
    except Exception as e:
        print("❌ Error al reproducir la voz:", e)

try:
    while True:
        # Leer fragmento de audio del micrófono
        pcm = np.frombuffer(
            mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16
        )

        # Alimentar al modelo de Wake Word
        prediction = oww_model.predict(pcm)

        # Guardar la confianza en una variable más limpia
        confianza = prediction["goo_stahv"]

        # --- RADAR DE CONFIANZA ---
        # Si escucha algo que se parece un poco (más de 20% pero menos de 65%)
        # usamos \r para que se actualice en la misma línea y no ensucie la consola
        if 0.1 < confianza <= 0.3:
            print(
                f"👂 Analizando... (Probabilidad de 'Gustav': {confianza:.2f})",
                end="\r",
                flush=True,
            )

        # Si la confianza supera el umbral del 65%
        if confianza > 0.2:
            print(f"\n🔔 ¡Wake Word detectado! (Confianza de disparo: {confianza:.2f})")

            # 1. Grabar lo que dices a continuación
            archivo = grabar_comando(duracion_segundos=5)

            # 2. PAUSAR el micrófono mientras transcribe para no acumular basura
            mic_stream.stop_stream()

            # 3. Transcribir y enviar al orquestador (Whisper imprimirá lo que entiende aquí)
            transcribir_y_enviar(archivo)

            # 4. Resetear la memoria interna del modelo Wake Word
            oww_model.reset()

            print("\n🟢 Volviendo a escuchar (Di 'Gustav')...")

            # 5. Reactivar el micrófono y vaciar cualquier residuo
            mic_stream.start_stream()
            mic_stream.read(
                mic_stream.get_read_available(), exception_on_overflow=False
            )

except KeyboardInterrupt:
    print("\n🛑 Apagando asistente...")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
