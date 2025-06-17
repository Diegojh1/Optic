#!/usr/bin/env python3
"""
Script de inicialización del Sistema de Traducción LSC
Este script configura automáticamente todo lo necesario para usar el sistema.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        logger.error("Se requiere Python 3.8 o superior")
        return False
    logger.info(f"Python {sys.version} - OK")
    return True

def install_requirements():
    """Instalar dependencias"""
    try:
        logger.info("Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al instalar dependencias: {e}")
        return False

def create_directories():
    """Crear directorios necesarios"""
    directories = [
        "data",
        "models",
        "database",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def initialize_database():
    """Inicializar base de datos con gestos LSC"""
    try:
        logger.info("Inicializando base de datos con gestos LSC...")
        from populate_lsc_database import main as populate_main
        populate_main()
        logger.info("Base de datos inicializada correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar base de datos: {e}")
        return False

def test_camera():
    """Probar acceso a la cámara"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("No se pudo acceder a la cámara. Verifica que esté conectada.")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            logger.info("Cámara funcionando correctamente")
            return True
        else:
            logger.warning("La cámara está disponible pero no funciona correctamente")
            return False
    except Exception as e:
        logger.error(f"Error al probar la cámara: {e}")
        return False

def test_system_components():
    """Probar componentes del sistema"""
    try:
        # Probar importación de módulos
        from database.gesture_db import GestureDatabase
        from models.gesture_model import GestureModel
        
        # Probar base de datos
        db = GestureDatabase()
        gestures = db.get_all_gestures()
        logger.info(f"Base de datos funcionando - {len(gestures)} gestos disponibles")
        
        # Probar MediaPipe
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        logger.info("MediaPipe funcionando correctamente")
        
        # Probar PyTorch
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch funcionando - Dispositivo: {device}")
        
        return True
    except Exception as e:
        logger.error(f"Error al probar componentes: {e}")
        return False

def show_usage_instructions():
    """Mostrar instrucciones de uso"""
    print("\n" + "="*60)
    print("🎉 ¡SISTEMA LSC CONFIGURADO EXITOSAMENTE! 🎉")
    print("="*60)
    print("\n📋 INSTRUCCIONES DE USO:")
    print("\n1. RECOLECTAR DATOS DE GESTOS:")
    print("   python collect_data.py")
    print("   - Presiona 'r' para grabar gestos")
    print("   - Presiona 'd' para eliminar la última secuencia")
    print("   - Presiona 'q' para salir")
    
    print("\n2. GESTIONAR GESTOS EN LA BASE DE DATOS:")
    print("   python manage_gestures.py --action list")
    print("   python manage_gestures.py --action search --query 'hola'")
    print("   python manage_gestures.py --action add_translation --gesture_id 1 --text 'saludo'")
    
    print("\n3. ENTRENAR EL MODELO:")
    print("   python train_model.py")
    print("   - Entrena el modelo con los datos recolectados")
    print("   - Guarda el modelo en models/gesture_model.pth")
    
    print("\n4. USAR EL TRADUCTOR EN TIEMPO REAL:")
    print("   python main.py")
    print("   - Traduce gestos LSC a texto en tiempo real")
    print("   - Presiona 'q' para salir")
    
    print("\n📊 INFORMACIÓN DEL SISTEMA:")
    try:
        from database.gesture_db import GestureDatabase
        db = GestureDatabase()
        gestures = db.get_all_gestures()
        total_sequences = sum(len(db.get_gesture_sequences(g["id"])) for g in gestures)
        print(f"   - Gestos disponibles: {len(gestures)}")
        print(f"   - Secuencias de entrenamiento: {total_sequences}")
        
        # Mostrar categorías
        categories = {}
        for gesture in gestures:
            category = gesture["category"]
            categories[category] = categories.get(category, 0) + 1
        
        print(f"   - Categorías: {', '.join(categories.keys())}")
    except:
        pass
    
    print("\n🆘 SOPORTE:")
    print("   - Verifica que la cámara esté conectada")
    print("   - Asegúrate de tener buena iluminación")
    print("   - Mantén las manos visibles en el encuadre")
    print("   - Consulta el README.md para más información")
    print("\n" + "="*60)

def main():
    """Función principal de configuración"""
    print("🚀 Iniciando configuración del Sistema LSC...")
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias
    if not install_requirements():
        logger.error("Falló la instalación de dependencias")
        sys.exit(1)
    
    # Inicializar base de datos
    if not initialize_database():
        logger.error("Falló la inicialización de la base de datos")
        sys.exit(1)
    
    # Probar cámara
    test_camera()
    
    # Probar componentes del sistema
    if not test_system_components():
        logger.error("Algunos componentes del sistema no funcionan correctamente")
        sys.exit(1)
    
    # Mostrar instrucciones
    show_usage_instructions()

if __name__ == "__main__":
    main() 