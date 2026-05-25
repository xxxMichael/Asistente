import pyaudio
import numpy as np
import requests
import openwakeword
from openwakeword.model import Model
import wave
import time
import asyncio
import edge_tts
import pygame
import os
import io
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE APIS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
N8N_WEBHOOK_URL = os.getenv("WEBHOOK_N8N_URL")

# 1. Configuración de Modelos
print("Cargando modelo Wake Word 'Gustav'...")
oww_model = Model(wakeword_models=["./models/goo_stahv.onnx"])

# Inicializamos el reproductor de audio
pygame.mixer.init()

# 2. Configuración del Micrófono (PyAudio)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

def grabar_comando_vad(umbral_silencio=450, espera_inicio_segundos=5.0, espera_fin_segundos=3.0):
    """Graba directamente en la memoria RAM sin usar el disco duro"""
    print(f"🎤 Escuchando... (Tienes {espera_inicio_segundos}s para empezar a hablar)")
    frames = []
    chunks_silencio = 0
    
    chunks_espera_inicio = int((RATE / CHUNK) * espera_inicio_segundos)
    chunks_espera_fin = int((RATE / CHUNK) * espera_fin_segundos)
    max_chunks_seguridad = int((RATE / CHUNK) * 60.0)
    
    chunks_grabados = 0
    ha_empezado_a_hablar = False

    while chunks_grabados < max_chunks_seguridad:
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        chunks_grabados += 1
        
        # Optimización: Cálculo de energía más ligero para el CPU
        pcm = np.frombuffer(data, dtype=np.int16)
        energia = np.abs(pcm.astype(np.int32)).mean() if len(pcm) > 0 else 0        
        
        if energia >= umbral_silencio:
            if not ha_empezado_a_hablar:
                print("🗣️ Voz detectada, grabando comando...")
                ha_empezado_a_hablar = True
            chunks_silencio = 0
        else:
            chunks_silencio += 1
            
        if not ha_empezado_a_hablar and chunks_silencio > chunks_espera_inicio:
            print("🛑 Tiempo de espera inicial agotado.")
            break
            
        elif ha_empezado_a_hablar and chunks_silencio > chunks_espera_fin:
            print("🛑 Fin del comando (Silencio detectado).")
            break

    # AQUÍ ESTÁ LA MAGIA: Guardamos en un buffer de memoria RAM, no en el disco
    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    
    audio_buffer.seek(0) # Reiniciamos el puntero del buffer para que Groq lo lea desde el principio
    return audio_buffer

def transcribir_con_groq(audio_buffer):
    """Envía el buffer de memoria directamente a Groq"""
    if audio_buffer.getbuffer().nbytes < 44:  # Si el buffer está vacío (solo cabecera WAV)
        return ""
        
    print("🧠 Transcribiendo con Groq (Whisper Large V3)...")
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    # Engañamos a Groq diciéndole que es un archivo .wav, pero le pasamos la RAM
    files = {
        "file": ("comando.wav", audio_buffer, "audio/wav")
    }
    data = {
        "model": "whisper-large-v3",
        "language": "es",
        "response_format": "json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            texto = response.json().get("text", "").strip()
            print(f"⚡ Transcrito en {time.time() - start_time:.2f}s")
            return texto
        else:
            print(f"❌ Error en Groq API: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print("❌ Error de conexión con la API de Groq:", e)
        return ""

def procesar_y_enviar_n8n(texto):
    if not texto:
        return

    print(f"✅ Entendido: '{texto}'")
    try:
        print("⏳ Esperando respuesta de Gustav...")
        respuesta = requests.post(N8N_WEBHOOK_URL, json={"texto": texto})
        
        if respuesta.status_code != 200:
            print(f"❌ Error interno en n8n (Código {respuesta.status_code}).")
            print(f"🔍 Detalles del servidor: {respuesta.text}")
            hablar("Hubo un problema procesando la solicitud en mi servidor.")
            return

        datos = respuesta.json()

        if isinstance(datos, dict) and "output" in datos:
            mensaje_ia = datos["output"]
        elif isinstance(datos, list) and len(datos) > 0 and "output" in datos[0]:
            mensaje_ia = datos[0]["output"]
        else:
            mensaje_ia = respuesta.text

        mensaje_ia = mensaje_ia.replace('*', '').replace('"', '').replace('`', '').replace('\n', ', ')
        print(f"\n🤖 Gustav dice: {mensaje_ia}\n")
        hablar(mensaje_ia)

    except ValueError:
        print(f"❌ n8n no devolvió un JSON válido. Respuesta en crudo: {respuesta.text}")
    except Exception as e:
        print("❌ Error de red al conectar con n8n:", e)

def hablar(texto):
    try:
        archivo_voz = "respuesta.mp3"
        voz = "es-EC-LuisNeural" 
        velocidad = "+20%" 
        
        async def generar_voz():
            comunicador = edge_tts.Communicate(texto, voice=voz, rate=velocidad)
            await comunicador.save(archivo_voz)
            
        asyncio.run(generar_voz())
        
        pygame.mixer.music.load(archivo_voz)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        pygame.mixer.music.unload()
        if os.path.exists(archivo_voz):
            os.remove(archivo_voz)
            
    except Exception as e:
        print("❌ Error en el sistema de audio de salida:", e)

# --- BUCLE PRINCIPAL ---
print("🟢 Sistema listo. Di 'Gustav' para empezar...")
try:
    while True:
        pcm = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        prediction = oww_model.predict(pcm)
        confianza = prediction["goo_stahv"]

        if 0 < confianza <= 0.05:
            print(f"👂 Analizando... (Probabilidad de 'Gustav': {confianza:.2f})", end="\r", flush=True)

        if confianza > 0.05:
            print(f"\n🔔 ¡Wake Word detectado! (Confianza: {confianza:.2f})")

            # Ahora recibimos un buffer de memoria en lugar de un nombre de archivo
            audio_buffer = grabar_comando_vad()
            mic_stream.stop_stream()

            texto_transcrito = transcribir_con_groq(audio_buffer)

            procesar_y_enviar_n8n(texto_transcrito)

            oww_model.reset()
            print("\n🟢 Volviendo a escuchar (Di 'Gustav')...")

            mic_stream.start_stream()
            mic_stream.read(mic_stream.get_read_available(), exception_on_overflow=False)

except KeyboardInterrupt:
    print("\n🛑 Apagando asistente...")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()