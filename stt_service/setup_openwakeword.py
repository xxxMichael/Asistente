#!/usr/bin/env python3
"""
Script para descargar los recursos necesarios de openwakeword
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Obtener la ruta del paquete openwakeword
import openwakeword
oww_path = Path(openwakeword.__file__).parent

resources_dir = oww_path / "resources" / "models"
resources_dir.mkdir(parents=True, exist_ok=True)

# URLs de los modelos necesarios
models_to_download = {
    "melspectrogram.onnx": "https://huggingface.co/raven-2/openwakeword/resolve/main/models/melspectrogram.onnx"
}

print(f"📦 Descargando recursos a: {resources_dir}")

for model_name, url in models_to_download.items():
    model_path = resources_dir / model_name
    
    if model_path.exists():
        print(f"✅ {model_name} ya existe")
        continue
    
    print(f"⬇️ Descargando {model_name}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ {model_name} descargado exitosamente")
    except Exception as e:
        print(f"❌ Error descargando {model_name}: {e}")
        sys.exit(1)

print("✅ Todos los recursos están listos!")
