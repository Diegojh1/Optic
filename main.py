import cv2
import mediapipe as mp
import time
import requests
import json
import threading
import queue
import base64
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
from typing import Dict, List, Tuple, Any, Optional
import os
import math
from scipy.spatial import distance
import pygame
import re
import difflib  # Para comparación de cadenas y fuzzy matching
import sys

# Añadir manejo de errores para la inicialización
def safe_init():
    """Inicialización segura con manejo de errores"""
    try:
        # Configuración de temas de la aplicación
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Inicializar pygame para efectos de sonido
        pygame.mixer.init()
        
        print("✅ Inicialización básica completada")
        return True
    except Exception as e:
        print(f"❌ Error en inicialización básica: {e}")
        return False

# Llamar a la inicialización segura
if not safe_init():
    print("Error crítico en la inicialización")
    input("Presiona Enter para salir...")
    sys.exit(1)

# Configuración de la API de Gemini
GEMINI_API_KEY = "AIzaSyDVjmUAxkg4GYmpi4IHkggDEyM-WLzXZa4"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
HEADERS = {'Content-Type': 'application/json'}

# Directorios para datos
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
GESTURES_DIR = os.path.join(DATA_DIR, "gestures")
CAPTURES_DIR = os.path.join(DATA_DIR, "captures")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
SOUNDS_DIR = os.path.join(DATA_DIR, "sounds")
GIFS_DIR = os.path.join(DATA_DIR, "gifs")  # Directorio para los GIFs de señas

# Crear directorios si no existen
def create_directories():
    """Crea los directorios necesarios"""
    directories = [DATA_DIR, CACHE_DIR, GESTURES_DIR, CAPTURES_DIR, EXPORTS_DIR, SOUNDS_DIR, GIFS_DIR]
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"✅ Directorio creado: {directory}")
        except Exception as e:
            print(f"❌ Error creando directorio {directory}: {e}")

# Crear directorios
create_directories()

# Inicializar MediaPipe con manejo de errores
def init_mediapipe():
    """Inicializa MediaPipe con manejo de errores"""
    try:
        global mp_hands, mp_face_mesh, mp_draw, hands, face_mesh
        global drawing_spec_face, drawing_spec_hands, connection_spec
        
        # Inicializar MediaPipe Hands y Face Mesh
        mp_hands = mp.solutions.hands
        mp_face_mesh = mp.solutions.face_mesh
        mp_draw = mp.solutions.drawing_utils

        # Configuraciones de detección
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Estilos de dibujo
        drawing_spec_face = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        drawing_spec_hands = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        connection_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1)
        
        print("✅ MediaPipe inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error inicializando MediaPipe: {e}")
        return False

# Inicializar MediaPipe
if not init_mediapipe():
    print("Error crítico: No se pudo inicializar MediaPipe")
    input("Presiona Enter para salir...")
    sys.exit(1)

# Variables para la detección de los ojos
eye_closed_time = 0
eye_closed_threshold = 1.0  # Duración para activar o desactivar el zoom
blink_counter = 0
last_blink_time = 0

# Índices de los ojos en MediaPipe Face Mesh
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Variables para el control de gestos y API
last_api_call_time = datetime.now()
api_cooldown = 2  # Tiempo mínimo entre llamadas a la API
gestures_queue = queue.Queue(maxsize=10)
last_gesture_time = datetime.now()
gesture_inactivity_threshold = 1.5
current_translation = ""
translation_history = []
max_history_size = 50
is_running = False
camera_active = False

# Variables de estabilización
last_translation_time = datetime.now()
translation_cooldown = 1.0  # Tiempo mínimo entre traducciones (1 segundo)
gesture_stability_buffer = []  # Buffer para estabilizar gestos
stability_buffer_size = 3  # Número de detecciones consecutivas para confirmar (reducido)
processing_frame_skip = 3  # Procesar cada N frames para mejorar rendimiento (más frecuente)
frame_counter = 0

# Variables para preparación de gestos (tiempos muy reducidos)
gesture_preparation_time = 0.3  # Tiempo para preparar el gesto antes de procesar (muy rápido)
hand_detection_start_time = None  # Momento cuando se detectaron manos por primera vez
gesture_ready_threshold = 0.5  # Tiempo que las manos deben estar estables antes de procesar (rápido)
last_hand_position = None  # Posición anterior de la mano para detectar estabilidad
hand_stability_threshold = 0.08  # Umbral de movimiento para considerar la mano estable (más tolerante)

# Variables de zoom
zoom_factor = 1.0
previous_thumb_index_distance = None
zoom_speed = 3.0
stability_threshold = 0.01
zoom_mode = False
zoom_smooth_factor = 0.3

# Configuración para la detección de movimiento
significant_movement = False
last_landmarks = None
movement_threshold = 0.02

# Configuración para el sistema
use_context = True
ai_precision_mode = "balanced"
confidence_threshold = 0.65
max_api_calls_per_minute = 15
api_calls_this_minute = 0
api_minute_start = datetime.now()
visualization_mode = "advanced"
show_metrics = True
system_status = "standby"
current_mode = "translation"

# Colores para la interfaz
COLOR_PRIMARY = (0, 200, 255)
COLOR_SECONDARY = (0, 120, 255)
COLOR_ACCENT = (255, 60, 0)
COLOR_SUCCESS = (0, 255, 120)
COLOR_WARNING = (255, 180, 0)
COLOR_ERROR = (255, 40, 40)
COLOR_BACKGROUND = (8, 12, 24)

# Cache para reducir llamadas a la API
translation_cache = {}
gesture_signature_cache = {}

# Configuración de efectos de sonido
sound_effects_enabled = True
sound_volume = 0.5

# Configuración de la cámara
camera_index = 0
camera_resolution = (640, 480)
frame_rate = 30

# Variables para el sistema de entrenamiento
TRAINING_MIN_SAMPLES = 10  # Número mínimo de muestras para entrenar
TRAINING_MAX_SAMPLES = 50  # Número máximo de muestras para entrenar
training_mode = False  # Indica si estamos en modo entrenamiento
current_gesture_name = ""  # Nombre del gesto que se está entrenando
training_samples = []  # Lista para almacenar las muestras de entrenamiento
training_counter = 0  # Contador de muestras capturadas
training_in_progress = False  # Indica si hay una sesión de entrenamiento activa
training_ready = False  # Indica si hay suficientes muestras para entrenar
training_model = None  # Modelo de entrenamiento actual

# Variables para gestos entrenados
trained_gestures = {}  # Diccionario con los gestos entrenados cargados
gesture_recognition_threshold = 0.7  # Umbral de similitud para reconocer gestos entrenados (más tolerante)
use_trained_gestures = True  # Usar gestos entrenados como prioridad
high_confidence_threshold = 0.6  # Umbral para alta confianza (uso directo) - más bajo
medium_confidence_threshold = 0.3  # Umbral para confianza media (verificar con IA) - más bajo
hybrid_mode = True  # Modo híbrido: fusiona gestos entrenados con IA

# =====================================================
# CONFIGURACIÓN DE GIFS PREDEFINIDOS PARA TEXTO A SEÑAS
# =====================================================

# Diccionario de GIFs para la traducción de texto a señas
# AQUI PUEDO INSERTAR MAS GIFS PARA QUE SEAN RECONOCIDOS
text_to_sign_dict = {
    # Formato: "palabra_clave": "nombre_archivo.gif"
    # Ejemplo:
    "buenos dias": "buenos dias.gif",
    "hola": "hola.gif",
    "gracias": "gracias.gif",
    
    # QUE NO SE ME OLVIDE AGREGAR MAS GIFS AQUI, EJEMPLO:
    # "palabra1": "archivo1.gif",
    # "palabra2": "archivo2.gif",
}

# Diccionario de sinónimos para facilitar la búsqueda de coincidencias
synonym_dict = {
    "hola": ["saludar", "saludos", "buenos días", "buen día", "qué tal", "hey"],
    "adios": ["chao", "hasta luego", "despedida", "nos vemos", "hasta pronto"],
    "gracias": ["agradecer", "agradecido", "gracie", "agradecimiento", "thank you"],
    "perdon": ["disculpa", "lo siento", "disculparse", "perdón", "sorry"],
    "si": ["afirmativo", "sí", "claro", "por supuesto", "ok", "afirmar"],
    "no": ["negativo", "negar", "negación", "nunca", "jamás"],
    "ayuda": ["auxilio", "socorro", "ayudar", "asistencia", "apoyo", "help"],
    "por_favor": ["por favor", "favor", "please", "amablemente"],
    "nombre": ["cómo te llamas", "cuál es tu nombre", "identificación", "identidad", "quién eres"],
    "amor": ["te quiero", "te amo", "cariño", "corazón", "querer", "amar"],
    "familia": ["parientes", "familiares", "papá", "mamá", "padres", "hermanos"],
    "comer": ["comida", "alimento", "alimentarse", "cenar", "almorzar", "desayunar"],
    "trabajar": ["trabajo", "labor", "empleo", "ocupación", "profesión"],
    "estudiar": ["estudio", "aprender", "aprendizaje", "educación"],
    
    # Puedo añadir mas sinonimos de esta forma:
    # "palabra_clave": ["sinónimo1", "sinónimo2", "sinónimo3"],
}

# Para almacenar las palabras similares procesadas
processed_words_dict = {}

# Cargar efectos de sonido con manejo de errores
def load_sound_effects():
    """Carga los efectos de sonido con manejo de errores"""
    global sound_effects
    sound_effects = {}
    
    sound_files = {
        "scan": "scan.mp3",
        "beep": "beep.mp3", 
        "success": "success.mp3",
        "alert": "alert.mp3",
        "error": "error.mp3",
        "click": "click.mp3",
        "capture": "capture.mp3"
    }
    
    for name, filename in sound_files.items():
        filepath = os.path.join(SOUNDS_DIR, filename)
        if os.path.exists(filepath):
            sound_effects[name] = filepath
        else:
            # Crear archivo de sonido dummy si no existe
            sound_effects[name] = None
            if name == "capture":
                print(f"⚠️  Archivo de sonido CAPTURE no encontrado: {filename}")
                print(f"   📁 Coloca el archivo en: {SOUNDS_DIR}")
            else:
                print(f"⚠️  Archivo de sonido no encontrado: {filename}")

# Cargar efectos de sonido
load_sound_effects()

# Inicializar palabras procesadas para búsqueda difusa
def initialize_processed_words():
    """Preprocesa las palabras para búsqueda rápida de similitudes"""
    global processed_words_dict
    processed_words_dict = {}
    
    # Obtener todas las palabras clave del diccionario
    all_keys = list(text_to_sign_dict.keys())
    
    # Procesar cada palabra
    for word in all_keys:
        # Normalizar: quitar acentos, convertir a minúsculas, quitar caracteres especiales
        normalized = normalize_text(word)
        processed_words_dict[normalized] = word
    
    # También procesar los sinónimos
    for key_word, synonyms in synonym_dict.items():
        for syn in synonyms:
            normalized = normalize_text(syn)
            processed_words_dict[normalized] = syn

# Normalización de texto para comparaciones
def normalize_text(text):
    """Normaliza un texto para facilitar comparaciones"""
    # Convertir a minúsculas
    text = text.lower()
    
    # Reemplazar acentos
    accent_map = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ñ': 'n'
    }
    
    for accented, normal in accent_map.items():
        text = text.replace(accented, normal)
    
    # Conservar solo letras y números, reemplazar otros caracteres con espacios
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text)
    
    # Quitar espacios al inicio y al final
    return text.strip()

# Función mejorada para encontrar la mejor coincidencia para una palabra
def find_best_match(word):
    """
    Encuentra la mejor coincidencia para una palabra en el diccionario de señas.
    Usa coincidencia exacta, sinónimos, palabras similares y fuzzy matching.
    """
    if not word or len(word.strip()) == 0:
        return None
    
    # Normalizar la entrada
    original_word = word
    word = normalize_text(word)
    
    # 1. Comprobar coincidencia exacta
    if word in text_to_sign_dict:
        return text_to_sign_dict[word]
    
    # Comprobar versión con guiones bajos
    word_with_underscores = word.replace(" ", "_")
    if word_with_underscores in text_to_sign_dict:
        return text_to_sign_dict[word_with_underscores]
    
    # 2. Buscar en palabras procesadas (coincidencias normalizadas)
    if word in processed_words_dict:
        processed_word = processed_words_dict[word]
        if processed_word in text_to_sign_dict:
            return text_to_sign_dict[processed_word]
    
    # 3. Comprobar si contiene alguna palabra clave
    best_match = None
    best_score = 0
    threshold = 0.6  # Umbral de confianza mínima
    
    # Primero, buscar palabras contenidas exactamente
    for key in text_to_sign_dict:
        if key in word or key.replace("_", "") in word.replace(" ", ""):
            # La palabra clave está contenida en la entrada
            match_score = len(key) / len(word)
            if match_score > best_score:
                best_score = match_score
                best_match = text_to_sign_dict[key]
    
    if best_score > threshold:
        return best_match
    
    # 4. Buscar en sinónimos
    for base_word, synonyms in synonym_dict.items():
        # Verificar si la palabra de entrada está entre los sinónimos
        for syn in synonyms:
            syn_normalized = normalize_text(syn)
            if word == syn_normalized or word in syn_normalized or syn_normalized in word:
                # Encontramos un sinónimo, buscar el GIF para la palabra base
                base_key = base_word.replace("_", " ").lower()
                if base_key in text_to_sign_dict:
                    return text_to_sign_dict[base_key]
                elif base_word in text_to_sign_dict:
                    return text_to_sign_dict[base_word]
    
    # 5. Usar fuzzy matching para encontrar palabras similares
    best_match = None
    best_ratio = threshold  # Umbral mínimo de similitud
    
    for key in text_to_sign_dict:
        # Calcular similitud usando SequenceMatcher
        ratio = difflib.SequenceMatcher(None, word, normalize_text(key)).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = text_to_sign_dict[key]
    
    # Si encontramos una coincidencia lo suficientemente buena
    if best_match:
        return best_match
    
    # No se encontraron coincidencias
    return None

# Función para reproducir efectos de sonido
def play_sound(sound_name):
    """Reproduce efectos de sonido con manejo de errores"""
    if not sound_effects_enabled:
        return
        
    try:
        if sound_name in sound_effects and sound_effects[sound_name] and os.path.exists(sound_effects[sound_name]):
            sound = pygame.mixer.Sound(sound_effects[sound_name])
            sound.set_volume(sound_volume)
            sound.play()
    except Exception as e:
        print(f"Error reproduciendo sonido {sound_name}: {e}")

# Función para calcular la relación de aspecto del ojo (EAR)
def calcular_ear(landmarks, eye_indices):
    """Calcula la relación de aspecto del ojo."""
    try:
        # Cálculo de distancias euclidianas entre puntos clave
        A = distance.euclidean(
            (landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y),
            (landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y)
        )
        B = distance.euclidean(
            (landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y),
            (landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y)
        )
        C = distance.euclidean(
            (landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y),
            (landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y)
        )
        
        # Evitar división por cero
        if C < 0.001:
            return 0
            
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        return 0

# Función para verificar el estado de los ojos
def check_eye_status(landmarks, frame=None):
    """Analiza el estado de los ojos con detección de parpadeo."""
    global eye_closed_time, zoom_mode, blink_counter, last_blink_time
    global training_mode, training_in_progress  # Añadir variables de entrenamiento
    
    try:
        # Calcular EAR (Eye Aspect Ratio) para ambos ojos
        left_eye_ear = calcular_ear(landmarks, left_eye_indices)
        right_eye_ear = calcular_ear(landmarks, right_eye_indices)
        
        # Promedio de EAR para reducir falsos positivos
        avg_ear = (left_eye_ear + right_eye_ear) / 2.0
        
        # Umbral para detección
        ear_threshold = 0.2
        
        current_time = time.time()
        
        # Detección de ojos cerrados (parpadeo)
        if avg_ear < ear_threshold:
            # Iniciar conteo si los ojos acaban de cerrarse
            if eye_closed_time == 0:
                eye_closed_time = current_time
                
            # Verificar si los ojos han estado cerrados suficiente tiempo
            elif current_time - eye_closed_time >= eye_closed_threshold:
                
                # MODO ENTRENAMIENTO: Capturar gesto por parpadeo
                if training_mode and training_in_progress:
                    print("👁️ Parpadeo detectado en modo entrenamiento - Capturando gesto...")
                    
                    # Llamar a la función de captura desde la aplicación
                    if hasattr(app, 'capture_gesture'):
                        app.capture_gesture()
                    
                    # Reproducir efecto de sonido específico para captura
                    play_sound("capture")
                    
                    # Restablecer el tiempo para evitar múltiples capturas
                    eye_closed_time = 0
                    
                    # Visualización en modo entrenamiento
                    if frame is not None:
                        cv2.putText(frame, "CAPTURA POR PARPADEO", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SUCCESS, 2)
                
                # MODO NORMAL: Toggle de zoom
                elif not training_mode:
                    # Toggle de zoom
                    zoom_mode = not zoom_mode
                    
                    # Reproducir efecto de sonido para feedback
                    play_sound("beep")
                    
                    # Restablecer el tiempo para evitar múltiples toggles
                    eye_closed_time = 0
                    
                    # Registrar el parpadeo largo como un comando
                    if current_time - last_blink_time < 2.0:
                        blink_counter += 1
                    else:
                        blink_counter = 1
                    
                    last_blink_time = current_time
                    
                    # Visualización en modo científico/debug
                    if frame is not None and visualization_mode == "scientific":
                        cv2.putText(frame, "COMANDO: ZOOM " + ("ON" if zoom_mode else "OFF"), 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
        else:
            # Resetear tiempo de ojos cerrados
            eye_closed_time = 0
            
        return {
            "left_ear": left_eye_ear,
            "right_ear": right_eye_ear,
            "avg_ear": avg_ear,
            "eyes_closed": avg_ear < ear_threshold,
            "blink_counter": blink_counter
        }
    except Exception as e:
        return {
            "left_ear": 0,
            "right_ear": 0,
            "avg_ear": 0,
            "eyes_closed": False,
            "blink_counter": 0
        }

# Función para detectar gestos y calcular distancias
def detect_movement(puntos_mano, frame=None):
    """Detecta movimientos significativos y calcula distancias entre dedos."""
    global previous_thumb_index_distance, zoom_factor, zoom_speed, last_landmarks
    global significant_movement, zoom_smooth_factor
    
    try:
        if not puntos_mano.landmark:
            return False
        
        landmarks = list(puntos_mano.landmark)
        
        # Calcular la distancia entre el pulgar y el índice (para el zoom)
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
        
        # Distancia euclidiana 3D para mayor precisión
        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Inicializar distancia previa si es necesario
        if previous_thumb_index_distance is None:
            previous_thumb_index_distance = thumb_index_distance
        
        # Ajustar el zoom con suavizado para movimientos más naturales
        if zoom_mode and abs(thumb_index_distance - previous_thumb_index_distance) > stability_threshold:
            # Calcular objetivo de zoom con limitaciones
            delta_zoom = (thumb_index_distance - previous_thumb_index_distance) * zoom_speed
            
            # Aplicar suavizado para evitar cambios bruscos
            target_zoom_factor = zoom_factor + delta_zoom
            
            # Suavizado de transición de zoom para mayor naturalidad
            zoom_factor = zoom_factor * (1 - zoom_smooth_factor) + target_zoom_factor * zoom_smooth_factor
            
            # Limitar el rango de zoom para usabilidad
            zoom_factor = max(1.0, min(zoom_factor, 30.0))
            
            previous_thumb_index_distance = thumb_index_distance
        
        # Detección de movimiento
        if last_landmarks is not None:
            # Calcular movimiento ponderando articulaciones importantes
            total_movement = 0
            weights = {
                4: 2.0,   # Punta del pulgar
                8: 2.0,   # Punta del índice
                12: 1.5,  # Punta del medio
                16: 1.0,  # Punta del anular
                20: 1.0,  # Punta del meñique
                0: 1.5,   # Muñeca
            }
            
            total_weight = sum(weights.values())
            
            # Calcular movimiento ponderado
            weighted_movement = 0
            for i, (current, previous) in enumerate(zip(landmarks, last_landmarks)):
                if i in weights:
                    # Vector de movimiento 3D
                    movement_vector = np.array([
                        current.x - previous.x,
                        current.y - previous.y,
                        current.z - previous.z
                    ])
                    
                    # Magnitud del movimiento
                    movement = np.linalg.norm(movement_vector)
                    
                    # Aplicar peso
                    weighted_movement += movement * weights[i]
            
            # Normalizar por la suma de pesos
            avg_movement = weighted_movement / total_weight
            
            # Umbral para movimiento significativo
            significant_movement = avg_movement > movement_threshold
            
            # Visualizar datos de movimiento en modo científico
            if frame is not None and visualization_mode == "scientific":
                cv2.putText(frame, f"Mov: {avg_movement:.4f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SUCCESS, 1)
        
        # Actualizar landmarks para la próxima iteración
        last_landmarks = landmarks.copy()
        
        return significant_movement
    except Exception as e:
        return False

# Función para extraer características de gestos como firma única
def extract_gesture_signature(landmarks):
    """Extrae una "firma" del gesto basada en posiciones relativas."""
    try:
        if not landmarks:
            return None
        
        # Obtener puntos clave de los dedos
        key_points = [4, 8, 12, 16, 20]  # Puntas de los dedos
        base_point = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])  # Muñeca
        
        # Extraer características: distancias relativas a la muñeca y ángulos
        signature = []
        
        # 1. Distancias relativas normalizadas
        for point_idx in key_points:
            point = np.array([landmarks[point_idx].x, landmarks[point_idx].y, landmarks[point_idx].z])
            # Normalizar por tamaño de la mano (distancia de muñeca a medio)
            hand_size = np.linalg.norm(
                np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z]) - base_point)
            
            if hand_size > 0:  # Evitar división por cero
                rel_dist = np.linalg.norm(point - base_point) / hand_size
                signature.append(rel_dist)
            else:
                signature.append(0)
        
        # 2. Ángulos entre dedos adyacentes
        for i in range(1, 5):  # Índice-medio, medio-anular, anular-meñique
            p1 = np.array([landmarks[key_points[i-1]].x, landmarks[key_points[i-1]].y])
            p2 = np.array([landmarks[key_points[i]].x, landmarks[key_points[i]].y])
            p0 = np.array([landmarks[0].x, landmarks[0].y])  # Muñeca como referencia
            
            v1 = p1 - p0
            v2 = p2 - p0
            
            # Calcular ángulo
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product > 0:
                cos_angle = dot_product / norm_product
                # Limitar a [-1, 1] para evitar errores numéricos
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.arccos(cos_angle)
                signature.append(angle)
            else:
                signature.append(0)
        
        # Convertir a una representación compacta (hash del gesto)
        signature_array = np.array(signature)
        # Redondear para reducir ruido
        signature_rounded = np.round(signature_array * 100) / 100
        # Convertir a tuple para hacerlo hashable
        return tuple(signature_rounded)
    
    except Exception as e:
        return None

# Cargar gestos entrenados
def load_trained_gestures():
    """Carga todos los gestos entrenados desde los archivos JSON."""
    global trained_gestures
    
    try:
        trained_gestures = {}
        
        if not os.path.exists(GESTURES_DIR):
            print("📁 Directorio de gestos no encontrado")
            return
        
        # Buscar todos los archivos JSON en el directorio de gestos
        gesture_files = [f for f in os.listdir(GESTURES_DIR) if f.endswith('.json')]
        
        if not gesture_files:
            print("📂 No se encontraron gestos entrenados")
            return
        
        print(f"📁 Encontrados {len(gesture_files)} archivos de gestos: {gesture_files}")
        
        for gesture_file in gesture_files:
            gesture_path = os.path.join(GESTURES_DIR, gesture_file)
            print(f"📂 Cargando archivo: {gesture_file}")
            
            try:
                with open(gesture_path, 'r', encoding='utf-8') as f:
                    gesture_data = json.load(f)
                
                gesture_name = gesture_data.get('name', '')
                samples = gesture_data.get('samples', [])
                
                print(f"   📊 Gesto: '{gesture_name}', Muestras: {len(samples)}")
                
                if gesture_name and samples:
                    # Procesar las muestras para crear firmas de referencia
                    gesture_signatures = []
                    
                    for sample_idx, sample in enumerate(samples):
                        print(f"      🔍 Procesando muestra {sample_idx + 1}/{len(samples)} (manos: {len(sample)})")
                        
                        for hand_idx, hand_data in enumerate(sample):
                            if hand_data and len(hand_data) >= 21:  # Verificar que hay datos suficientes
                                # Convertir lista de vuelta a formato de landmarks simulados
                                landmarks = []
                                for point_idx, point in enumerate(hand_data):
                                    if len(point) >= 3:  # Verificar que tiene x, y, z
                                        # Crear objeto tipo landmark simulado
                                        landmark = type('Landmark', (), {
                                            'x': point[0], 'y': point[1], 'z': point[2]
                                        })()
                                        landmarks.append(landmark)
                                
                                if len(landmarks) >= 21:  # MediaPipe requiere 21 landmarks por mano
                                    # Extraer firma del gesto
                                    signature = extract_gesture_signature(landmarks)
                                    if signature:
                                        gesture_signatures.append(signature)
                                        print(f"         ✅ Firma extraída: {signature[:3]}... (primeros 3 valores)")
                                    else:
                                        print(f"         ❌ No se pudo extraer firma")
                                else:
                                    print(f"         ⚠️  Insuficientes landmarks: {len(landmarks)}/21")
                            else:
                                print(f"         ⚠️  Datos de mano insuficientes: {len(hand_data) if hand_data else 0} puntos")
                    
                    if gesture_signatures:
                        trained_gestures[gesture_name] = {
                            'signatures': gesture_signatures,
                            'name': gesture_name,
                            'sample_count': len(samples)
                        }
                        print(f"✅ Gesto cargado: '{gesture_name}' ({len(gesture_signatures)} firmas válidas)")
                    else:
                        print(f"❌ No se pudieron extraer firmas válidas para '{gesture_name}'")
                else:
                    print(f"❌ Datos inválidos en {gesture_file}: nombre='{gesture_name}', muestras={len(samples)}")
            
            except Exception as e:
                print(f"❌ Error cargando gesto {gesture_file}: {e}")
        
        print(f"📚 Resumen: {len(trained_gestures)} gestos entrenados cargados correctamente")
        for name, data in trained_gestures.items():
            print(f"   🎯 {name}: {data['sample_count']} muestras originales → {len(data['signatures'])} firmas")
        
    except Exception as e:
        print(f"❌ Error general cargando gestos entrenados: {e}")

# Comparar gesto actual con gestos entrenados
def compare_with_trained_gestures(current_landmarks):
    """Compara el gesto actual con los gestos entrenados."""
    global trained_gestures
    
    try:
        if not use_trained_gestures or not trained_gestures or not current_landmarks:
            if not use_trained_gestures:
                print("🔍 Gestos entrenados desactivados")
            elif not trained_gestures:
                print("🔍 No hay gestos entrenados cargados")
            else:
                print("🔍 No hay landmarks para comparar")
            return None
        
        # Extraer firma del gesto actual
        current_signature = extract_gesture_signature(current_landmarks)
        if not current_signature:
            print("🔍 No se pudo extraer firma del gesto actual")
            return None
        
        print(f"🔍 Comparando gesto actual con {len(trained_gestures)} gestos entrenados...")
        print(f"🔍 Firma actual: {current_signature[:5]}... (primeros 5 valores)")
        
        best_match = None
        best_score = float('inf')
        all_distances = {}
        
        # Comparar con cada gesto entrenado
        for gesture_name, gesture_data in trained_gestures.items():
            signatures = gesture_data['signatures']
            
            min_distance_for_gesture = float('inf')
            
            # Encontrar la mejor coincidencia dentro de este gesto
            for i, signature in enumerate(signatures):
                try:
                    # Calcular distancia euclidiana entre firmas
                    if len(current_signature) == len(signature):
                        distance = np.linalg.norm(
                            np.array(current_signature) - np.array(signature)
                        )
                        
                        if distance < min_distance_for_gesture:
                            min_distance_for_gesture = distance
                        
                        if distance < best_score:
                            best_score = distance
                            best_match = gesture_name
                
                except Exception as e:
                    continue
            
            all_distances[gesture_name] = min_distance_for_gesture
        
        # Verificar si la mejor coincidencia supera el umbral
        if best_match and best_score < gesture_recognition_threshold:
            confidence = max(0, 1 - (best_score / gesture_recognition_threshold))
            print(f"🎯 ¡GESTO RECONOCIDO! '{best_match}' (confianza: {confidence:.1%}, distancia: {best_score:.3f})")
            return {
                'gesture': best_match,
                'confidence': confidence,
                'distance': best_score
            }
        
        return None
    
    except Exception as e:
        print(f"❌ Error comparando gestos: {e}")
        return None

# Sistema de preparación y estabilidad de gestos
def check_hand_stability(hand_landmarks):
    """
    Verifica si la mano está suficientemente estable para procesar el gesto.
    """
    global last_hand_position, hand_detection_start_time, gesture_ready_threshold
    
    try:
        if not hand_landmarks or len(hand_landmarks) == 0:
            # No hay manos, resetear
            last_hand_position = None
            hand_detection_start_time = None
            return False, "sin_manos"
        
        current_time = datetime.now()
        
        # Calcular posición promedio de la mano (centro de la palma)
        landmarks = hand_landmarks[0].landmark
        hand_center = np.array([
            landmarks[9].x,  # Centro de la palma
            landmarks[9].y,
            landmarks[9].z
        ])
        
        # Si es la primera detección de manos
        if hand_detection_start_time is None:
            hand_detection_start_time = current_time
            last_hand_position = hand_center
            return False, "detectando_manos"
        
        # Verificar si han pasado suficientes segundos desde la primera detección
        time_since_detection = (current_time - hand_detection_start_time).total_seconds()
        
        if time_since_detection < gesture_preparation_time:
            # Aún en período de preparación
            remaining_time = gesture_preparation_time - time_since_detection
            return False, f"preparando ({remaining_time:.1f}s)"
        
        # Verificar estabilidad de la mano
        if last_hand_position is not None:
            movement = np.linalg.norm(hand_center - last_hand_position)
            
            if movement > hand_stability_threshold:
                # Mano se está moviendo mucho, resetear timer de estabilidad
                hand_detection_start_time = current_time
                last_hand_position = hand_center
                return False, "mano_en_movimiento"
            
            # Verificar si la mano ha estado estable suficiente tiempo
            if time_since_detection >= gesture_ready_threshold:
                last_hand_position = hand_center
                return True, "listo_para_procesar"
        
        # Actualizar posición
        last_hand_position = hand_center
        return False, "estabilizando"
        
    except Exception as e:
        print(f"❌ Error verificando estabilidad: {e}")
        return False, "error"

def reset_hand_stability():
    """Resetea el sistema de estabilidad de manos."""
    global hand_detection_start_time, last_hand_position
    hand_detection_start_time = None
    last_hand_position = None

# Sistema de estabilización de gestos
def stabilize_gesture_recognition(gesture_result):
    """
    Estabiliza el reconocimiento de gestos usando un buffer de detecciones consecutivas.
    """
    global gesture_stability_buffer, last_translation_time, translation_cooldown
    
    try:
        current_time = datetime.now()
        
        # Verificar cooldown de traducciones
        if (current_time - last_translation_time).total_seconds() < translation_cooldown:
            return None
        
        # Agregar resultado al buffer
        if gesture_result:
            gesture_stability_buffer.append({
                'gesture': gesture_result['translation'],
                'confidence': gesture_result['confidence'],
                'source': gesture_result['source'],
                'timestamp': current_time
            })
        else:
            # Si no hay gesto, agregar None
            gesture_stability_buffer.append(None)
        
        # Mantener el tamaño del buffer
        if len(gesture_stability_buffer) > stability_buffer_size:
            gesture_stability_buffer.pop(0)
        
        # Verificar si tenemos suficientes detecciones
        if len(gesture_stability_buffer) < stability_buffer_size:
            return None
        
        # Contar gestos válidos en el buffer
        valid_gestures = [g for g in gesture_stability_buffer if g is not None]
        
        if len(valid_gestures) < 2:  # Necesitamos al menos 2 detecciones válidas
            return None
        
        # Verificar consistencia del gesto
        gesture_names = [g['gesture'] for g in valid_gestures]
        most_common_gesture = max(set(gesture_names), key=gesture_names.count)
        
        # Contar cuántas veces aparece el gesto más común
        count = gesture_names.count(most_common_gesture)
        
        # Si el gesto aparece en al menos el 60% de las detecciones
        if count >= len(valid_gestures) * 0.6:
            # Calcular confianza promedio
            confidences = [g['confidence'] for g in valid_gestures if g['gesture'] == most_common_gesture]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Obtener fuente más reciente
            recent_source = valid_gestures[-1]['source']
            
            print(f"🎯 Gesto estabilizado: '{most_common_gesture}' ({count}/{len(valid_gestures)} detecciones)")
            
            # Limpiar buffer después de confirmación
            gesture_stability_buffer.clear()
            last_translation_time = current_time
            
            return {
                'translation': most_common_gesture,
                'confidence': avg_confidence,
                'source': f'{recent_source} (estabilizado)',
                'detections': count
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error en estabilización: {e}")
        return None

# Sistema híbrido de reconocimiento
def hybrid_gesture_recognition(frame, hand_landmarks, face_landmarks=None):
    """
    Sistema híbrido que prioriza fuertemente gestos entrenados sobre IA.
    """
    global current_translation, translation_history
    global api_calls_this_minute, api_minute_start
    
    try:
        current_time = datetime.now()
        
        # Verificar límite de API por minuto
        if (current_time - api_minute_start).total_seconds() >= 60:
            api_calls_this_minute = 0
            api_minute_start = current_time
        
        trained_result = None
        final_result = None
        
        # PASO 1: PRIORIDAD ABSOLUTA a gestos entrenados
        if use_trained_gestures and hand_landmarks and len(hand_landmarks) > 0:
            trained_result = compare_with_trained_gestures(hand_landmarks[0].landmark)
        
        if trained_result:
            confidence = trained_result['confidence']
            gesture_name = trained_result['gesture']
            
            # NUEVA LÓGICA: Si hay cualquier resultado entrenado, usarlo directamente
            if confidence >= medium_confidence_threshold:  # Umbral muy bajo (0.3)
                print(f"🎯 GESTO ENTRENADO DETECTADO: '{gesture_name}' ({confidence:.1%}) - USANDO SIN IA")
                final_result = {
                    'translation': gesture_name,
                    'source': f'entrenado {confidence:.1%}',
                    'confidence': confidence
                }
            else:
                # Solo si la confianza es muy muy baja, intentar IA como respaldo
                print(f"🤔 CONFIANZA MUY BAJA - Gesto: '{gesture_name}' ({confidence:.1%})")
                if api_calls_this_minute < max_api_calls_per_minute:
                    print("🤖 Consultando IA como último recurso...")
                    ai_result = call_gemini_api_direct(frame, hand_landmarks, face_landmarks)
                    if ai_result:
                        final_result = {
                            'translation': ai_result,
                            'source': 'IA (respaldo)',
                            'confidence': 0.6
                        }
                    else:
                        # Si IA falla, usar el resultado entrenado de todas formas
                        final_result = {
                            'translation': gesture_name,
                            'source': f'entrenado {confidence:.1%} (IA falló)',
                            'confidence': confidence
                        }
                else:
                    # Sin API disponible, usar resultado entrenado obligatoriamente
                    final_result = {
                        'translation': gesture_name,
                        'source': f'entrenado {confidence:.1%} (sin API)',
                        'confidence': confidence
                    }
        else:
            # PASO 2: Solo usar IA si NO hay resultado entrenado en absoluto
            if api_calls_this_minute < max_api_calls_per_minute:
                print("🤖 Sin gesto entrenado, usando IA...")
                ai_result = call_gemini_api_direct(frame, hand_landmarks, face_landmarks)
                if ai_result:
                    final_result = {
                        'translation': ai_result,
                        'source': 'IA',
                        'confidence': 0.7
                    }
            else:
                print("❌ Sin gestos entrenados y sin API disponible")
                return False
        
        # PASO 3: Aplicar resultado final
        if final_result:
            current_translation = final_result['translation']
            
            # Añadir a historial con información de fuente
            timestamp = current_time.strftime("%H:%M:%S")
            translation_entry = f"[{timestamp}] {current_translation} ({final_result['source']})"
            translation_history.append(translation_entry)
            
            # Mantener tamaño máximo del historial
            if len(translation_history) > max_history_size:
                translation_history.pop(0)
            
            # Reproducir sonido de éxito
            play_sound("success")
            
            print(f"✅ TRADUCCIÓN: '{current_translation}' - {final_result['source']}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error en reconocimiento híbrido: {e}")
        return False

# Función auxiliar para llamada rápida a IA
def call_gemini_api_fast(frame, hand_landmarks, face_landmarks=None):
    """Versión rápida de la llamada a IA para verificación."""
    try:
        # Usar configuración rápida
        img_base64 = frame_to_base64(frame, quality=70, max_size=600)
        
        prompt = """
        Identifica rápidamente el gesto de lengua de señas en la imagen.
        SOLO responde con UNA PALABRA que describa el gesto.
        Si no hay gesto claro, responde: NO_GESTURE_DETECTED
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }}
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 5,
                "topK": 20,
                "topP": 0.8
            }
        }
        
        response = requests.post(GEMINI_URL, headers=HEADERS, json=payload, timeout=3)
        
        if response.status_code == 200:
            response_data = response.json()
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                text = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
                text = text.strip('"\'.,;:!?()[]{}').strip()
                
                if text != "NO_GESTURE_DETECTED" and "no se detecta" not in text.lower():
                    return text
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error en llamada rápida IA: {e}")
        return None

# Función auxiliar para llamada directa a IA
def call_gemini_api_direct(frame, hand_landmarks, face_landmarks=None):
    """Llamada directa a IA sin verificaciones previas."""
    global api_calls_this_minute
    
    try:
        api_calls_this_minute += 1
        return call_gemini_api_fast(frame, hand_landmarks, face_landmarks)
    except Exception as e:
        print(f"❌ Error en llamada directa IA: {e}")
        return None

# Convertir frame a base64 para enviar a la API
def frame_to_base64(frame, quality=80, max_size=1024):
    """Convierte el frame a base64 con optimizaciones de tamaño y calidad."""
    try:
        # Reducir tamaño si es necesario
        h, w = frame.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        # Codificar con calidad optimizada
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        # Fallback a calidad inferior en caso de error
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return base64.b64encode(buffer).decode('utf-8')

# Extraer información relevante de los puntos
def extract_hand_data(landmarks, add_derivatives=True):
    """Extrae datos detallados de los landmarks de las manos."""
    hand_data = []
    
    try:
        if landmarks:
            # Datos de posición
            for i, landmark in enumerate(landmarks.landmark):
                point_data = {
                    "index": i,
                    "x": round(landmark.x, 5),
                    "y": round(landmark.y, 5),
                    "z": round(landmark.z, 5)
                }
                
                # Agregar datos de movimiento si se solicita
                if add_derivatives and last_landmarks is not None and i < len(last_landmarks):
                    # Calcular velocidad (derivada de posición)
                    point_data["dx"] = round(landmark.x - last_landmarks[i].x, 5)
                    point_data["dy"] = round(landmark.y - last_landmarks[i].y, 5)
                    point_data["dz"] = round(landmark.z - last_landmarks[i].z, 5)
                
                hand_data.append(point_data)
    except Exception as e:
        pass
        
    return hand_data

# Función para llamar a Gemini API
def call_gemini_api(frame, hand_landmarks, face_landmarks=None):
    """
    Llama a la API de Gemini para interpretar gestos de lengua de señas.
    """
    global current_translation, translation_history
    global api_calls_this_minute, api_minute_start
    global translation_cache, gesture_signature_cache
    
    try:
        # Verificar límite de API por minuto
        current_time = datetime.now()
        
        # Reiniciar contador si ha pasado un minuto
        if (current_time - api_minute_start).total_seconds() >= 60:
            api_calls_this_minute = 0
            api_minute_start = current_time
        
        # Verificar si hemos alcanzado el límite
        if api_calls_this_minute >= max_api_calls_per_minute:
            return False
        
        # NUEVA FUNCIONALIDAD: Verificar primero los gestos entrenados
        if hand_landmarks and len(hand_landmarks) > 0:
            trained_result = compare_with_trained_gestures(hand_landmarks[0].landmark)
            if trained_result:
                # Se encontró coincidencia con gesto entrenado
                current_translation = trained_result['gesture']
                
                # Añadir a historial con timestamp y marca de entrenado
                timestamp = current_time.strftime("%H:%M:%S")
                confidence_text = f"(entrenado - {trained_result['confidence']:.1%})"
                translation_entry = f"[{timestamp}] {current_translation} {confidence_text}"
                translation_history.append(translation_entry)
                
                # Mantener tamaño máximo del historial
                if len(translation_history) > max_history_size:
                    translation_history.pop(0)
                
                # Reproducir sonido de éxito
                play_sound("success")
                
                return True
        
        # Optimización: Comprobar el caché de traducciones
        gesture_signature = None
        if hand_landmarks and len(hand_landmarks) > 0:
            # Extraer firma del gesto primario
            gesture_signature = extract_gesture_signature(hand_landmarks[0].landmark)
        # Verificar si tenemos esta traducción en caché
            if gesture_signature in translation_cache:
                cached_translation = translation_cache[gesture_signature]
                
                # Usar la traducción en caché si es reciente (menos de 10 minutos)
                cache_time = gesture_signature_cache.get(gesture_signature, datetime.min)
                if (current_time - cache_time).total_seconds() < 600:  # 10 minutos
                    current_translation = cached_translation
                    
                    # Añadir a historial con timestamp
                    timestamp = current_time.strftime("%H:%M:%S")
                    translation_entry = f"[{timestamp}] {current_translation} (caché)"
                    translation_history.append(translation_entry)
                    
                    # Mantener tamaño máximo del historial
                    if len(translation_history) > max_history_size:
                        translation_history.pop(0)
                    
                    # Reproducir sonido de éxito
                    play_sound("success")
                    
                    return True
        
        # Preparar imagen para la API
        img_base64 = frame_to_base64(frame, quality=85, max_size=800)
        
        # Preparar datos de manos
        hand_data = []
        if hand_landmarks:
            for hand in hand_landmarks:
                hand_data.append(extract_hand_data(hand))
        
        # Preparar datos de rostro si están disponibles
        face_data = []
        if face_landmarks:
            for face in face_landmarks:
                # Extraer puntos clave del rostro relevantes para LSC
                face_key_points = [
                    {"type": "left_eye", "indices": left_eye_indices},
                    {"type": "right_eye", "indices": right_eye_indices},
                    {"type": "mouth", "indices": [61, 291, 0, 17, 269, 405]}
                ]
                
                for feature in face_key_points:
                    points = []
                    for idx in feature["indices"]:
                        points.append({
                            "index": idx,
                            "x": face.landmark[idx].x,
                            "y": face.landmark[idx].y,
                            "z": face.landmark[idx].z
                        })
                    
                    face_data.append({
                        "feature": feature["type"],
                        "points": points
                    })
        
        # Preparar contexto de traducciones previas
        context_data = []
        if use_context and translation_history:
            context_data = translation_history[-3:]  # Usar hasta 3 traducciones previas
        
        # Crear prompt para Gemini
        if ai_precision_mode == "fast":
            temperature = 0.7
            max_output_tokens = 10
            instruction_detail = "breve"
        elif ai_precision_mode == "precise":
            temperature = 0.2
            max_output_tokens = 30
            instruction_detail = "detallado"
        else:  # "balanced"
            temperature = 0.4
            max_output_tokens = 20
            instruction_detail = "equilibrado"
        
        prompt = f"""
        Eres un intérprete experto de Lengua de Señas Colombiana (LSC) con una base de datos integrada.
        Analiza la imagen proporcionada y los datos de seguimiento de manos para identificar el gesto o seña LSC.
        
        Nivel de análisis: {instruction_detail}
        
        Analiza exhaustivamente:
        1. La posición y forma de las manos
        2. Las expresiones faciales (si son visibles)
        3. El movimiento implícito por las posiciones
        4. La relación entre las manos y el rostro (si es relevante)
        
        {"Contexto de gestos previos para referencia: " + ", ".join(context_data) if context_data else ""}
        
        Identifica con precisión el gesto LSC actual.
        
        SOLO RESPONDE CON LA TRADUCCIÓN DEL GESTO.
        Si no puedes identificar un gesto LSC claro, responde exactamente: "NO_GESTURE_DETECTED"
        No incluyas explicaciones, comillas ni puntuación adicional.
        """
        
        # Construir la solicitud para la API
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }},
                    {"text": f"Datos de puntos de mano: {json.dumps(hand_data)}"}
                ]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topK": 40,
                "topP": 0.95
            }
        }
        
        # Si hay datos faciales, añadirlos
        if face_data:
            payload["contents"][0]["parts"].append(
                {"text": f"Datos faciales: {json.dumps(face_data)}"}
            )
        
        # Realizar la llamada API
        response = requests.post(GEMINI_URL, headers=HEADERS, json=payload, timeout=5)
        
        # Incrementar contador de llamadas API
        api_calls_this_minute += 1
        
        # Procesar respuesta
        if response.status_code == 200:
            response_data = response.json()
            
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                text = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Comprobar si se detectó un gesto
                if text != "NO_GESTURE_DETECTED" and "no se detecta" not in text.lower() and "no puedo" not in text.lower():
                    # Limpiar el texto (eliminar puntuación y normalizar)
                    text = text.strip('"\'.,;:!?()[]{}').strip()
                    
                    # Actualizar la traducción actual
                    current_translation = text
                    
                    # Guardar en caché para futuras referencias
                    if gesture_signature:
                        translation_cache[gesture_signature] = text
                        gesture_signature_cache[gesture_signature] = current_time
                    
                    # Añadir a historial con timestamp
                    timestamp = current_time.strftime("%H:%M:%S")
                    translation_entry = f"[{timestamp}] {current_translation}"
                    translation_history.append(translation_entry)
                    
                    # Mantener tamaño máximo del historial
                    if len(translation_history) > max_history_size:
                        translation_history.pop(0)
                    
                    # Reproducir sonido de éxito
                    play_sound("success")
                    
                    return True
        
        # Si llegamos aquí, hubo un error o no se detectó un gesto
        return False
    
    except Exception as e:
        print(f"Error al llamar a la API: {e}")
        return False

# Función para añadir HUD a la imagen
def add_hud(frame, metrics=None):
    """Añade elementos de HUD al frame de video."""
    try:
        height, width = frame.shape[:2]
        
        # Marco tecnológico con esquinas
        corner_size = min(width, height) // 15
        line_thickness = 2
        
        # Colores para el marco
        primary_color = COLOR_PRIMARY
        
        # Esquina superior izquierda
        cv2.line(frame, (0, corner_size), (corner_size, 0), primary_color, line_thickness)
        cv2.line(frame, (0, 0), (corner_size, 0), primary_color, line_thickness)
        cv2.line(frame, (0, 0), (0, corner_size), primary_color, line_thickness)
        
        # Esquina superior derecha
        cv2.line(frame, (width-corner_size, 0), (width, corner_size), primary_color, line_thickness)
        cv2.line(frame, (width-corner_size, 0), (width, 0), primary_color, line_thickness)
        cv2.line(frame, (width, 0), (width, corner_size), primary_color, line_thickness)
        
        # Esquina inferior izquierda
        cv2.line(frame, (0, height-corner_size), (corner_size, height), primary_color, line_thickness)
        cv2.line(frame, (0, height), (corner_size, height), primary_color, line_thickness)
        cv2.line(frame, (0, height-corner_size), (0, height), primary_color, line_thickness)
        
        # Esquina inferior derecha
        cv2.line(frame, (width-corner_size, height), (width, height-corner_size), primary_color, line_thickness)
        cv2.line(frame, (width-corner_size, height), (width, height), primary_color, line_thickness)
        cv2.line(frame, (width, height-corner_size), (width, height), primary_color, line_thickness)
        
        # Panel de métricas en tiempo real
        if show_metrics and metrics:
            # Fondo para el panel de métricas
            metrics_panel_height = 120 if visualization_mode == "scientific" else 80
            metrics_panel_width = 200
            panel_x = width - metrics_panel_width - 10
            panel_y = 10
            
            # Crear panel con efecto de transparencia
            cv2.rectangle(frame, 
                         (panel_x, panel_y), 
                         (panel_x + metrics_panel_width, panel_y + metrics_panel_height), 
                         (20, 20, 40), 
                         -1)  # Relleno
            
            # Borde del panel
            cv2.rectangle(frame, 
                         (panel_x, panel_y), 
                         (panel_x + metrics_panel_width, panel_y + metrics_panel_height), 
                         primary_color, 
                         1)
            
            # Título del panel
            cv2.putText(frame, 
                       "SISTEMA MÉTRICAS", 
                       (panel_x + 10, panel_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (0, 200, 255), 
                       1)
            
            # Métricas con valores
            y_offset = panel_y + 35
            line_spacing = 15
            
            # Mostrar métricas disponibles
            for label, value in metrics.items():
                if isinstance(value, float):
                    value_text = f"{value:.2f}"
                else:
                    value_text = str(value)
                
                metric_text = f"{label}: {value_text}"
                cv2.putText(frame, 
                           metric_text, 
                           (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, 
                           (180, 180, 220), 
                           1)
                y_offset += line_spacing
        
        # Indicador de estado del sistema
        status_colors = {
            "standby": (100, 100, 200),
            "analyzing": (0, 200, 255),
            "translating": (0, 255, 100),
            "learning": (255, 100, 0),
            "error": (255, 0, 0)
        }
        
        status_text = system_status.upper()
        status_color = status_colors.get(system_status.lower(), (200, 200, 200))
        
        # Indicador de estado
        status_x = 20
        status_y = 30
        
        # Fondo del indicador
        cv2.rectangle(frame, 
                     (status_x - 10, status_y - 20), 
                     (status_x + 150, status_y + 5), 
                     (20, 20, 40), 
                     -1)
        
        # Texto de estado
        cv2.putText(frame, 
                   f"ESTADO: {status_text}", 
                   (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   status_color, 
                   1)
        
        # Indicador de modo
        mode_colors = {
            "translation": (0, 200, 255),
            "text_to_sign": (255, 100, 100),
        }
        
        mode_text = current_mode.upper()
        mode_color = mode_colors.get(current_mode.lower(), (200, 200, 200))
        
        # Indicador de modo
        mode_x = 20
        mode_y = 60
        
        # Fondo del indicador
        cv2.rectangle(frame, 
                     (mode_x - 10, mode_y - 20), 
                     (mode_x + 150, mode_y + 5), 
                     (20, 20, 40), 
                     -1)
        
        # Texto de modo
        cv2.putText(frame, 
                   f"MODO: {mode_text}", 
                   (mode_x, mode_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   mode_color, 
                   1)
        
        # Indicador de cooldown de traducción
        global last_translation_time, translation_cooldown
        cooldown_remaining = translation_cooldown - (datetime.now() - last_translation_time).total_seconds()
        if cooldown_remaining > 0:
            cooldown_x = 20
            cooldown_y = 90
            
            # Fondo del indicador
            cv2.rectangle(frame, 
                         (cooldown_x - 10, cooldown_y - 20), 
                         (cooldown_x + 200, cooldown_y + 5), 
                         (20, 20, 40), 
                         -1)
            
            # Texto de cooldown
            cv2.putText(frame, 
                       f"COOLDOWN: {cooldown_remaining:.1f}s", 
                       (cooldown_x, cooldown_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 180, 0), 
                       1)
        
        # Indicador de zoom (solo si está activo)
        if zoom_mode:
            zoom_x = 20
            zoom_y = 120 if cooldown_remaining > 0 else 90
            
            # Fondo del indicador
            cv2.rectangle(frame, 
                         (zoom_x - 10, zoom_y - 20), 
                         (zoom_x + 150, zoom_y + 5), 
                         (20, 20, 40), 
                         -1)
            
            # Texto de zoom
            cv2.putText(frame, 
                       f"ZOOM: {zoom_factor:.1f}x", 
                       (zoom_x, zoom_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 180, 0), 
                       1)
        
        return frame
    
    except Exception as e:
        print(f"Error al añadir HUD: {e}")
        return frame

# Función para añadir panel de traducción
def add_translation_panel(frame, translation_text):
    """Añade un panel para mostrar la traducción."""
    try:
        height, width = frame.shape[:2]
        
        # Si no hay traducción, no añadir panel
        if not translation_text:
            return frame
        
        # Tamaño del panel
        panel_height = 100
        panel_y = height - panel_height
        
        # Crear copia para el panel
        overlay = frame.copy()
        
        # Fondo del panel semitransparente
        cv2.rectangle(overlay, 
                     (0, panel_y), 
                     (width, height), 
                     (8, 12, 24), 
                     -1)
        
        # Aplicar el panel al frame con transparencia
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Añadir bordes
        cv2.line(frame, (0, panel_y), (width, panel_y), COLOR_PRIMARY, 2)
        
        # Añadir etiqueta "TRADUCCIÓN"
        label_x = 20
        label_y = panel_y + 25
        
        cv2.putText(frame, "TRADUCCIÓN:", 
                   (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   COLOR_PRIMARY, 
                   2)
        
        # Añadir texto de traducción
        text_x = 20
        text_y = panel_y + 65
        
        # Texto principal
        cv2.putText(frame, translation_text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, 
                   (240, 240, 240), 
                   2)
        
        return frame
    
    except Exception as e:
        print(f"Error al añadir panel de traducción: {e}")
        return frame

# Función principal para capturar video y procesar
def video_stream():
    """Función principal para capturar y procesar el video en tiempo real."""
    global current_translation, last_gesture_time
    global last_api_call_time, camera_active, system_status
    global resultados_manos, resultados_rostro  # Añadir variables globales
    global frame_counter  # Añadir contador de frames
    
    if not camera_active:
        return
    
    try:
        # Abrir la cámara con resolución específica
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Error: No se puede acceder a la cámara")
            system_status = "error"
            camera_active = False
            return
        
        # Configurar propiedades de la cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, frame_rate)
        
        # Variables para FPS
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Reproducir sonido de inicio
        play_sound("scan")
        
        while camera_active:
            # Calcular FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 1.0:  # Actualizar FPS cada segundo
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Capturar frame
            success, img = cap.read()
            if not success:
                break
            
            # Incrementar contador de frames
            frame_counter += 1
            
            # Estado predeterminado según el modo
            system_status = "learning" if training_mode else "analyzing"
            
            # Convertir a RGB para MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detección de manos y rostro (siempre necesaria para visualización)
            resultados_manos = hands.process(img_rgb)
            resultados_rostro = face_mesh.process(img_rgb)
            
            # Crear métricas para mostrar
            metrics = {
                "FPS": round(fps, 1),
                "Zoom": round(zoom_factor, 1),
                "API/min": api_calls_this_minute,
                "Muestras": training_counter if training_mode else len(translation_history)
            }
            
            # Verificar actividad de gestos
            current_time = datetime.now()
            is_active = False
            
            # Procesar resultados del rostro
            if resultados_rostro.multi_face_landmarks:
                for face_idx, landmarks in enumerate(resultados_rostro.multi_face_landmarks):
                    # Dibujar landmarks faciales con estilo según el modo
                    if visualization_mode in ["advanced", "scientific"]:
                        # Dibujar puntos clave de los ojos
                        for eye_indices in [left_eye_indices, right_eye_indices]:
                            for i, idx in enumerate(eye_indices):
                                # Coordenadas del punto
                                x = int(landmarks.landmark[idx].x * img.shape[1])
                                y = int(landmarks.landmark[idx].y * img.shape[0])
                                
                                # Dibujar punto
                                cv2.circle(img, (x, y), 2, COLOR_PRIMARY, -1)
                    
                    # Verificar estado de los ojos para comandos
                    check_eye_status(landmarks.landmark, img)
            
            # Procesar resultados de manos
            if resultados_manos.multi_hand_landmarks:
                for hand_idx, puntos_mano in enumerate(resultados_manos.multi_hand_landmarks):
                    # Visualizar manos
                    if visualization_mode != "minimal":
                        # Estilo de visualización de manos
                        mp_draw.draw_landmarks(
                            img, 
                            puntos_mano, 
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec_hands,
                            connection_drawing_spec=connection_spec
                        )
                        
                        # Añadir indicadores de puntas de dedos
                        for tip_idx in [4, 8, 12, 16, 20]:  # Índices de puntas de dedos
                            x = int(puntos_mano.landmark[tip_idx].x * img.shape[1])
                            y = int(puntos_mano.landmark[tip_idx].y * img.shape[0])
                            
                            # Dibujar círculo en la punta
                            cv2.circle(img, (x, y), 8, COLOR_ACCENT, -1)
                    
                    # Detectar movimiento significativo
                    if detect_movement(puntos_mano):
                        is_active = True
                        last_gesture_time = current_time
                
                # Verificar estabilidad de manos antes de procesar
                is_stable, stability_status = check_hand_stability(resultados_manos.multi_hand_landmarks)
                
                # Mostrar estado de preparación en la imagen
                status_text = f"Estado: {stability_status}"
                cv2.putText(img, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           COLOR_SUCCESS if is_stable else COLOR_WARNING, 2)
                
                # Procesamiento de reconocimiento solo si la mano está estable
                if is_stable and is_active and frame_counter % processing_frame_skip == 0:
                    if training_mode:
                        # En modo entrenamiento, actualizar la vista previa
                        system_status = "learning"
                    elif (current_time - last_api_call_time).total_seconds() > api_cooldown:
                        # En modo normal, usar sistema híbrido
                        system_status = "translating"
                        
                        print(f"🎯 Procesando gesto - Mano estable detectada")
                        hybrid_gesture_recognition(img.copy(), 
                                                 resultados_manos.multi_hand_landmarks,
                                                 resultados_rostro.multi_face_landmarks)
                        last_api_call_time = current_time
                        
                        # Resetear estabilidad después de procesar
                        reset_hand_stability()
            else:
                # No hay manos detectadas, resetear sistema de estabilidad
                reset_hand_stability()
            
            # Comprobar si ha pasado suficiente tiempo sin actividad
            if (current_time - last_gesture_time).total_seconds() > gesture_inactivity_threshold:
                if training_mode:
                    system_status = "learning"
                else:
                    system_status = "standby"
            
            # Aplicar zoom si está activado
            if zoom_mode and zoom_factor > 1.0:
                height, width = img.shape[:2]
                # Calcular las nuevas dimensiones
                new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
                
                # Aplicar interpolación para el zoom
                img_resized = cv2.resize(img, (new_width, new_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
                
                # Calcular las coordenadas de recorte para mantener el centro
                start_x = max(0, (new_width - width) // 2)
                start_y = max(0, (new_height - height) // 2)
                
                # Recortar la imagen para mantener las dimensiones originales
                if new_width > width and new_height > height:
                    img = img_resized[start_y:start_y + height, start_x:start_x + width]
            
            # Añadir HUD 
            img = add_hud(img, metrics)
            
            # Añadir panel de traducción si hay una traducción y no estamos en modo entrenamiento
            if current_translation and not training_mode:
                img = add_translation_panel(img, current_translation)
            
            # Convertir la imagen para mostrarla en la interfaz
            img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_display)
            
            # Redimensionar para ajustar al marco de la interfaz
            target_frame = app.training_video_frame if training_mode else app.video_frame
            if hasattr(app, 'training_video_frame' if training_mode else 'video_frame'):
                target_width = target_frame.winfo_width() - 20
                target_height = target_frame.winfo_height() - 20
                
                if target_width > 0 and target_height > 0:
                    # Mantener la relación de aspecto
                    img_width, img_height = img_pil.size
                    ratio = min(target_width/img_width, target_height/img_height)
                    new_size = (int(img_width*ratio), int(img_height*ratio))
                    
                    # Redimensionar
                    img_pil = img_pil.resize(new_size, Image.LANCZOS)
                    
                    # Convertir a formato compatible con tkinter
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # Actualizar la etiqueta de video según el modo
                    if training_mode:
                        if hasattr(app, 'training_video_label'):
                            app.training_video_label.configure(image=img_tk)
                            app.training_video_label.image = img_tk
                    else:
                        if hasattr(app, 'video_label'):
                            app.video_label.configure(image=img_tk)
                            app.video_label.image = img_tk
            
            # Actualizar el estado e información en tiempo real
            if hasattr(app, 'status_label'):
                mode_text = "ENTRENAMIENTO" if training_mode else "TRADUCCIÓN"
                status_text = f"Estado: Sistema {system_status.upper()} | Modo: {mode_text}"
                if zoom_mode:
                    status_text += f" | Zoom: {zoom_factor:.1f}x"
                app.status_label.configure(text=status_text)
            
            if hasattr(app, 'current_translation_label'):
                if training_mode:
                    if training_in_progress:
                        app.current_translation_label.configure(
                            text=f"Entrenando gesto: {current_gesture_name} ({training_counter}/{TRAINING_MIN_SAMPLES})"
                        )
                    else:
                        app.current_translation_label.configure(
                            text="Modo entrenamiento activo"
                        )
                elif current_translation:
                    app.current_translation_label.configure(
                        text=f"Traducción: {current_translation}"
                    )
                else:
                    app.current_translation_label.configure(
                        text="Esperando gestos de LSC..."
                    )
            
            # Procesar eventos de la interfaz
            if hasattr(app, 'update'):
                app.update()
            
            # Si se pide salir, romper el bucle
            if not camera_active:
                break
            
            # Control de velocidad de fotogramas para no sobrecargar la CPU
            if fps > frame_rate + 5:  # Si FPS es demasiado alto
                time.sleep(0.01)  # Pequeña pausa
        
        # Reproducir sonido de finalización
        play_sound("beep")
        
        # Liberar la cámara al finalizar
        cap.release()
        
    except Exception as e:
        print(f"Error en video_stream: {e}")
        system_status = "error"
        
        # Reproducir sonido de error
        play_sound("error")
        
        # Intentar liberar la cámara en caso de error
        try:
            cap.release()
        except:
            pass
        
        camera_active = False

# Hilo para procesar llamadas a la API en segundo plano
def api_worker():
    """Procesa las solicitudes de traducción en un hilo separado."""
    global is_running
    while is_running:
        try:
            # Obtener datos de la cola con timeout para poder revisar is_running
            data = gestures_queue.get(timeout=1)
            
            if data:
                frame, hand_landmarks, face_landmarks = data
                
                # Llamar a la API de Gemini
                call_gemini_api(frame, hand_landmarks, face_landmarks)
                
                # Marcar tarea como terminada
                gestures_queue.task_done()
                
                # Esperar para evitar exceder el límite de API
                # Tiempo dinámico basado en el número de llamadas por minuto
                sleep_time = max(60 / max_api_calls_per_minute, 2)
                time.sleep(sleep_time)
        
        except queue.Empty:
            # Cola vacía, solo esperar un poco
            pass
        
        except Exception as e:
            print(f"Error en el hilo de la API: {e}")
            time.sleep(1)  # Esperar un poco en caso de error
        
        # Verificar si debemos terminar
        if not is_running:
            break

# Clase principal de la aplicación
class SignLanguageTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configurar la ventana principal
        self.title("Traductor LSC Bidireccional")
        self.geometry("1280x720")
        self.minsize(1024, 700)
        
        # Establecer tema oscuro
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Crear fuentes personalizadas
        self.title_font = ("Helvetica", 18, "bold")
        self.subtitle_font = ("Helvetica", 14, "bold")
        self.text_font = ("Helvetica", 12)
        
        # Configurar el grid principal
        self.grid_columnconfigure(0, weight=0, minsize=280)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Contenido principal
        self.grid_rowconfigure(0, weight=1)
        
        # Crear marcos principales
        self.create_sidebar()
        self.create_main_content()
        
        # Variable para almacenar el GIF actual en modo texto a seña
        self.current_gif = None
        self.gif_frames = []
        self.current_frame_index = 0
        
        # Inicializar sistema de búsqueda de palabras similares
        initialize_processed_words()
        
        # Reproducir sonido de inicio
        play_sound("beep")
    
    def create_sidebar(self):
        """Crea la barra lateral con navegación y controles."""
        # Marco para el sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#101825")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(0, weight=0)  # Logo
        self.sidebar_frame.grid_rowconfigure(1, weight=0)  # Título
        self.sidebar_frame.grid_rowconfigure(2, weight=0)  # Botones
        self.sidebar_frame.grid_rowconfigure(3, weight=1)  # Espacio
        self.sidebar_frame.grid_rowconfigure(4, weight=0)  # Estado
        self.sidebar_frame.grid_propagate(False)
        
        # Logo
        self.logo_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.logo_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        self.logo_label = ctk.CTkLabel(
            self.logo_frame,text="LSC NEURAL", 
            font=("Helvetica", 24, "bold"),
            text_color="#00c3ff"
        )
        self.logo_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Título de la aplicación
        self.app_title = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Traductor Bidireccional\nde Lengua de Señas Colombiana",
            font=self.title_font,
            text_color="#ffffff"
        )
        self.app_title.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Marco para los botones
        self.menu_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#0c1420", corner_radius=10)
        self.menu_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Botones del menú con iconos
        self.btn_signs_to_text = ctk.CTkButton(
            self.menu_frame,
            text="🖐️ Señas a Texto",
            command=self.switch_to_signs_to_text,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_signs_to_text.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_text_to_signs = ctk.CTkButton(
            self.menu_frame,
            text="📝 Texto a Señas",
            command=self.switch_to_text_to_signs,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_text_to_signs.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_training = ctk.CTkButton(
            self.menu_frame,
            text="🎯 Entrenar Gestos",
            command=self.switch_to_training,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_training.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_history = ctk.CTkButton(
            self.menu_frame,
            text="📊 Historial",
            command=self.show_history,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_history.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_settings = ctk.CTkButton(
            self.menu_frame,
            text="⚙️ Configuración",
            command=self.show_settings,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_settings.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_about = ctk.CTkButton(
            self.menu_frame,
            text="ℹ️ Acerca de",
            command=self.show_about,
            font=("Helvetica", 13),
            height=40,
            fg_color="#10253a",
            hover_color="#173a5a",
            corner_radius=8
        )
        self.btn_about.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        # Marco para el estado actual
        self.status_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#0c1420", corner_radius=10)
        self.status_frame.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Etiquetas de estado
        self.status_label = ctk.CTkLabel(
            self.status_frame, 
            text="Estado: Sistema STANDBY | Zoom Inactivo",
            font=("Helvetica", 12),
            text_color="#00c3ff"
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.current_translation_label = ctk.CTkLabel(
            self.status_frame, 
            text="Esperando entrada...",
            font=("Helvetica", 12, "bold"),
            wraplength=240,
            text_color="#ffffff"
        )
        self.current_translation_label.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="w")
    
    def create_main_content(self):
        """Crea el contenido principal con frames para cada sección."""
        # Marco principal de contenido
        self.content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#15202b")
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Diccionario para almacenar los frames
        self.frames = {}
        
        # Crear frames para cada sección
        self.create_signs_to_text_frame()
        self.create_text_to_signs_frame()
        self.create_training_frame()  # Añadir frame de entrenamiento
        self.create_history_frame()
        self.create_settings_frame()
        self.create_about_frame()
        
        # Mostrar frame de señas a texto por defecto
        self.show_frame("signs_to_text")
    
    def create_signs_to_text_frame(self):
        """Crea el frame para traducción de señas a texto."""
        # Frame de señas a texto
        self.frames["signs_to_text"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["signs_to_text"].grid(row=0, column=0, sticky="nsew")
        self.frames["signs_to_text"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["signs_to_text"].grid_rowconfigure(1, weight=1)  # Video
        self.frames["signs_to_text"].grid_rowconfigure(2, weight=0)  # Controles
        self.frames["signs_to_text"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.signs_to_text_title_frame = ctk.CTkFrame(self.frames["signs_to_text"], fg_color="#101825", corner_radius=10)
        self.signs_to_text_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.signs_to_text_title = ctk.CTkLabel(
            self.signs_to_text_title_frame, 
            text="🔍 TRADUCCIÓN DE SEÑAS A TEXTO",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.signs_to_text_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Subtítulo con información de uso
        self.signs_to_text_subtitle = ctk.CTkLabel(
            self.signs_to_text_title_frame,
            text="Cierra los ojos durante 1 segundo para activar/desactivar el zoom | Gestiona el zoom separando pulgar e índice",
            font=("Helvetica", 11),
            text_color="#ffffff"
        )
        self.signs_to_text_subtitle.grid(row=1, column=0, padx=10, pady=(0, 10))
        
        # Marco para el video
        self.video_frame = ctk.CTkFrame(
            self.frames["signs_to_text"], 
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.video_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.video_frame.grid_propagate(False)
        
        # Etiqueta para mostrar el video
        self.video_label = ctk.CTkLabel(self.video_frame, text="Presiona 'Iniciar Cámara' para comenzar",
                                      font=("Helvetica", 16))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Panel de controles
        self.controls_frame = ctk.CTkFrame(
            self.frames["signs_to_text"], 
            fg_color="#101825",
            corner_radius=10
        )
        self.controls_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Botones de control
        self.btn_start = ctk.CTkButton(
            self.controls_frame, 
            text="▶️ INICIAR CÁMARA",
            command=self.start_camera,
            font=("Helvetica", 13, "bold"),
            fg_color="#004080",
            hover_color="#005fa0",
            height=45,
            corner_radius=8
        )
        self.btn_start.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_stop = ctk.CTkButton(
            self.controls_frame, 
            text="⏹️ DETENER CÁMARA",
            command=self.stop_camera,
            font=("Helvetica", 13, "bold"),
            fg_color="#802000",
            hover_color="#a03000",
            height=45,
            corner_radius=8,
            state="disabled"
        )
        self.btn_stop.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_screenshot = ctk.CTkButton(
            self.controls_frame, 
            text="📷 CAPTURAR IMAGEN",
            command=self.take_screenshot,
            font=("Helvetica", 13, "bold"),
            fg_color="#006060",
            hover_color="#008080",
            height=45,
            corner_radius=8
        )
        self.btn_screenshot.grid(row=0, column=2, padx=10, pady=10)
        
        self.btn_export = ctk.CTkButton(
            self.controls_frame, 
            text="💾 EXPORTAR DATOS",
            command=self.export_data,
            font=("Helvetica", 13, "bold"),
            fg_color="#404060",
            hover_color="#505080",
            height=45,
            corner_radius=8
        )
        self.btn_export.grid(row=0, column=3, padx=10, pady=10)
    
    def create_text_to_signs_frame(self):
        """Crea el frame para traducción de texto a señas."""
        # Frame de texto a señas
        self.frames["text_to_signs"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["text_to_signs"].grid(row=0, column=0, sticky="nsew")
        self.frames["text_to_signs"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["text_to_signs"].grid_rowconfigure(1, weight=0)  # Entrada
        self.frames["text_to_signs"].grid_rowconfigure(2, weight=1)  # Visualización
        self.frames["text_to_signs"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.text_to_signs_title_frame = ctk.CTkFrame(self.frames["text_to_signs"], fg_color="#101825", corner_radius=10)
        self.text_to_signs_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.text_to_signs_title = ctk.CTkLabel(
            self.text_to_signs_title_frame, 
            text="📝 TRADUCCIÓN DE TEXTO A SEÑAS",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.text_to_signs_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Subtítulo con información de uso 
        self.text_to_signs_subtitle = ctk.CTkLabel(
            self.text_to_signs_title_frame,
            text="Escribe cualquier palabra o frase para ver su traducción en señas - El sistema reconoce sinónimos y palabras similares",
            font=("Helvetica", 11),
            text_color="#ffffff"
        )
        self.text_to_signs_subtitle.grid(row=1, column=0, padx=10, pady=(0, 10))
        
        # Marco para entrada de texto
        self.text_input_frame = ctk.CTkFrame(
            self.frames["text_to_signs"],
            fg_color="#0a1020",
            corner_radius=10
        )
        self.text_input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.text_input_frame.grid_columnconfigure(0, weight=1)
        self.text_input_frame.grid_columnconfigure(1, weight=0)
        
        # Campo de entrada de texto
        self.text_input = ctk.CTkEntry(
            self.text_input_frame,
            placeholder_text="Escribe una palabra o frase para traducir a lengua de señas...",
            font=("Helvetica", 14),
            height=40,
            fg_color="#101830",
            text_color="#ffffff",
            border_color="#00a0e0",
            border_width=2
        )
        self.text_input.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Botón de traducir
        self.translate_btn = ctk.CTkButton(
            self.text_input_frame,
            text="Traducir",
            command=self.translate_text_to_sign,
            font=("Helvetica", 13, "bold"),
            fg_color="#004080",
            hover_color="#005fa0",
            width=100,
            height=40,
            corner_radius=8
        )
        self.translate_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Marco para mostrar el GIF
        self.gif_frame = ctk.CTkFrame(
            self.frames["text_to_signs"],
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.gif_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.gif_frame.grid_rowconfigure(0, weight=0)  # Título
        self.gif_frame.grid_rowconfigure(1, weight=1)  # GIF
        
        # Título del GIF
        self.gif_title = ctk.CTkLabel(
            self.gif_frame,
            text="Traducción en Señas",
            font=("Helvetica", 16, "bold"),
            text_color="#00c3ff"
        )
        self.gif_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Etiqueta para mostrar el GIF
        self.gif_label = ctk.CTkLabel(
            self.gif_frame,
            text="Escribe una palabra o frase para ver su traducción en señas",
            font=("Helvetica", 14),
            wraplength=500
        )
        self.gif_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
    
    def create_history_frame(self):
        """Crea el frame de historial con las traducciones realizadas."""
        # Frame de historial
        self.frames["history"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["history"].grid(row=0, column=0, sticky="nsew")
        self.frames["history"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["history"].grid_rowconfigure(1, weight=1)  # Contenido
        self.frames["history"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.history_title_frame = ctk.CTkFrame(self.frames["history"], fg_color="#101825", corner_radius=10)
        self.history_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.history_title = ctk.CTkLabel(
            self.history_title_frame, 
            text="📊 REGISTRO DE TRADUCCIONES",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.history_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Marco para el historial
        self.history_content_frame = ctk.CTkFrame(
            self.frames["history"], 
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.history_content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Área de texto para el historial
        self.history_text = ctk.CTkTextbox(
            self.history_content_frame,
            fg_color="#0c1420",
            text_color="#e0e0e0",
            font=("Helvetica", 12),
            corner_radius=10
        )
        self.history_text.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Inicializar el historial
        self.history_text.insert("1.0", "🔍 HISTORIAL DE TRADUCCIONES\n")
        self.history_text.insert("2.0", "══════════════════════════\n\n")
        self.history_text.insert("4.0", "No hay traducciones recientes.\n")
        
        # Hacer de solo lectura
        self.history_text.configure(state="disabled")
    
    def create_settings_frame(self):
        """Crea el frame de configuración con opciones ajustables."""
        # Frame de configuración
        self.frames["settings"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["settings"].grid(row=0, column=0, sticky="nsew")
        self.frames["settings"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["settings"].grid_rowconfigure(1, weight=1)  # Contenido
        self.frames["settings"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.settings_title_frame = ctk.CTkFrame(self.frames["settings"], fg_color="#101825", corner_radius=10)
        self.settings_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.settings_title = ctk.CTkLabel(
            self.settings_title_frame, 
            text="⚙️ CONFIGURACIÓN DEL SISTEMA",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.settings_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Marco para la configuración
        self.settings_content_frame = ctk.CTkFrame(
            self.frames["settings"], 
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.settings_content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.settings_content_frame.grid_columnconfigure(0, weight=1)
        
        # Crear un panel desplazable para la configuración
        self.settings_scroll = ctk.CTkScrollableFrame(
            self.settings_content_frame,
            fg_color="transparent",
            corner_radius=0
        )
        self.settings_scroll.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.settings_scroll.grid_columnconfigure(0, weight=1)
        
        # Sección: Configuración de detección
        self.create_settings_section(
            self.settings_scroll, 
            "🖐️ CONFIGURACIÓN DE DETECCIÓN",
            0
        )
        
        # Opciones de movimiento
        self.movement_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.movement_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.movement_label = ctk.CTkLabel(
            self.movement_frame, 
            text="Umbral de Movimiento:",
            font=("Helvetica", 12)
        )
        self.movement_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.movement_slider = ctk.CTkSlider(
            self.movement_frame,
            from_=0.01,
            to=0.05,
            number_of_steps=40,
            command=self.update_movement_threshold
        )
        self.movement_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.movement_slider.set(movement_threshold)
        
        self.movement_value = ctk.CTkLabel(
            self.movement_frame,
            text=f"{movement_threshold:.3f}",
            width=60
        )
        self.movement_value.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # Opciones de zoom
        self.zoom_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.zoom_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.zoom_label = ctk.CTkLabel(
            self.zoom_frame, 
            text="Velocidad de Zoom:",
            font=("Helvetica", 12)
        )
        self.zoom_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.zoom_slider = ctk.CTkSlider(
            self.zoom_frame,
            from_=1.0,
            to=5.0,
            number_of_steps=40,
            command=self.update_zoom_speed
        )
        self.zoom_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.zoom_slider.set(zoom_speed)
        
        self.zoom_value = ctk.CTkLabel(
            self.zoom_frame,
            text=f"{zoom_speed:.1f}",
            width=60
        )
        self.zoom_value.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # Sección: Configuración de API
        self.create_settings_section(
            self.settings_scroll, 
            "🌐 CONFIGURACIÓN DE API",
            3
        )
        
        # Opciones de API
        self.api_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.api_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        self.api_label = ctk.CTkLabel(
            self.api_frame, 
            text="Tiempo entre Llamadas (s):",
            font=("Helvetica", 12)
        )
        self.api_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.api_slider = ctk.CTkSlider(
            self.api_frame,
            from_=1.0,
            to=5.0,
            number_of_steps=40,
            command=self.update_api_cooldown
        )
        self.api_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.api_slider.set(api_cooldown)
        
        self.api_value = ctk.CTkLabel(
            self.api_frame,
            text=f"{api_cooldown:.1f}",
            width=60
        )
        self.api_value.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # Opciones de modo de IA
        self.ai_mode_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.ai_mode_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        self.ai_mode_label = ctk.CTkLabel(
            self.ai_mode_frame, 
            text="Modo de Precisión:",
            font=("Helvetica", 12)
        )
        self.ai_mode_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Variable para almacenar el modo seleccionado
        self.ai_mode_var = tk.StringVar(value=ai_precision_mode)
        
        # Opciones de modo
        self.ai_mode_option = ctk.CTkOptionMenu(
            self.ai_mode_frame,
            values=["fast", "balanced", "precise"],
            command=self.update_ai_mode,
            variable=self.ai_mode_var,
            width=150
        )
        self.ai_mode_option.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Sección: Configuración de Audio
        self.create_settings_section(
            self.settings_scroll, 
            "🔊 CONFIGURACIÓN DE AUDIO",
            6
        )
        
        # Opciones de sonido
        self.sound_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.sound_frame.grid(row=7, column=0, padx=20, pady=10, sticky="ew")
        
        # Variable para almacenar el estado del checkbox
        self.sound_var = tk.IntVar(value=1 if sound_effects_enabled else 0)
        
        self.sound_checkbox = ctk.CTkCheckBox(
            self.sound_frame,
            text="Activar Efectos de Sonido",
            command=self.toggle_sound,
            variable=self.sound_var,
            font=("Helvetica", 12)
        )
        self.sound_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Sección: Configuración de Gestos Entrenados
        self.create_settings_section(
            self.settings_scroll, 
            "🎯 CONFIGURACIÓN DE GESTOS ENTRENADOS",
            8
        )
        
        # Opciones de gestos entrenados
        self.trained_gestures_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.trained_gestures_frame.grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        
        # Variable para activar/desactivar gestos entrenados
        self.trained_gestures_var = tk.IntVar(value=1 if use_trained_gestures else 0)
        
        self.trained_gestures_checkbox = ctk.CTkCheckBox(
            self.trained_gestures_frame,
            text="Usar Gestos Entrenados",
            command=self.toggle_trained_gestures,
            variable=self.trained_gestures_var,
            font=("Helvetica", 12)
        )
        self.trained_gestures_checkbox.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Sensibilidad de reconocimiento
        self.sensitivity_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.sensitivity_frame.grid(row=10, column=0, padx=20, pady=10, sticky="ew")
        
        self.sensitivity_label = ctk.CTkLabel(
            self.sensitivity_frame, 
            text="Sensibilidad de Reconocimiento:",
            font=("Helvetica", 12)
        )
        self.sensitivity_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.sensitivity_slider = ctk.CTkSlider(
            self.sensitivity_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=self.update_gesture_threshold
        )
        self.sensitivity_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.sensitivity_slider.set(gesture_recognition_threshold)
        
        self.sensitivity_value = ctk.CTkLabel(
            self.sensitivity_frame,
            text=f"{gesture_recognition_threshold:.2f}",
            width=60
        )
        self.sensitivity_value.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # Cooldown de traducciones
        self.cooldown_frame = ctk.CTkFrame(self.settings_scroll, fg_color="#101830")
        self.cooldown_frame.grid(row=11, column=0, padx=20, pady=10, sticky="ew")
        
        self.cooldown_label = ctk.CTkLabel(
            self.cooldown_frame, 
            text="Tiempo entre Traducciones (s):",
            font=("Helvetica", 12)
        )
        self.cooldown_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.cooldown_slider = ctk.CTkSlider(
            self.cooldown_frame,
            from_=0.5,
            to=3.0,
            number_of_steps=25,
            command=self.update_translation_cooldown
        )
        self.cooldown_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.cooldown_slider.set(translation_cooldown)
        
        self.cooldown_value = ctk.CTkLabel(
            self.cooldown_frame,
            text=f"{translation_cooldown:.1f}",
            width=60
        )
        self.cooldown_value.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # Botón para recargar gestos
        self.reload_gestures_btn = ctk.CTkButton(
            self.sensitivity_frame,
            text="🔄 Recargar Gestos",
            command=self.reload_trained_gestures,
            font=("Helvetica", 12),
            fg_color="#006060",
            hover_color="#008080",
            height=30,
            corner_radius=5
        )
        self.reload_gestures_btn.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        
        # Botón para guardar configuración
        self.save_config_btn = ctk.CTkButton(
            self.settings_scroll,
            text="💾 GUARDAR CONFIGURACIÓN",
            command=self.save_config,
            font=("Helvetica", 14, "bold"),
            fg_color="#004080",
            hover_color="#005fa0",
            height=50,
            corner_radius=8
        )
        self.save_config_btn.grid(row=12, column=0, padx=20, pady=30)

    def create_about_frame(self):
        """Crea el frame de 'Acerca de' con información del sistema."""
        # Frame de Acerca de
        self.frames["about"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["about"].grid(row=0, column=0, sticky="nsew")
        self.frames["about"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["about"].grid_rowconfigure(1, weight=1)  # Contenido
        self.frames["about"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.about_title_frame = ctk.CTkFrame(self.frames["about"], fg_color="#101825", corner_radius=10)
        self.about_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.about_title = ctk.CTkLabel(
            self.about_title_frame, 
            text="ℹ️ ACERCA DEL SISTEMA",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.about_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Marco para la información
        self.about_content_frame = ctk.CTkFrame(
            self.frames["about"], 
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.about_content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.about_content_frame.grid_columnconfigure(0, weight=1)
        self.about_content_frame.grid_rowconfigure((0, 1, 2), weight=0)
        self.about_content_frame.grid_rowconfigure(3, weight=1)
        
        # Logo del sistema
        self.about_logo_frame = ctk.CTkFrame(self.about_content_frame, fg_color="#0c1420")
        self.about_logo_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.about_logo = ctk.CTkLabel(
            self.about_logo_frame,
            text="LSC NEURAL TRANSLATOR",
            font=("Helvetica", 24, "bold"),
            text_color="#00c3ff"
        )
        self.about_logo.grid(row=0, column=0, padx=20, pady=20)
        
        # Versión y fecha
        self.about_version_frame = ctk.CTkFrame(self.about_content_frame, fg_color="#0c1420")
        self.about_version_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.about_version = ctk.CTkLabel(
            self.about_version_frame,
            text="Versión 2.5.0 - Abril 2025",
            font=("Helvetica", 14),
            text_color="#ffffff"
        )
        self.about_version.grid(row=0, column=0, padx=20, pady=10)
        
        # Información técnica
        self.tech_frame = ctk.CTkFrame(self.about_content_frame, fg_color="#0c1420")
        self.tech_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        
        self.tech_title = ctk.CTkLabel(
            self.tech_frame,
            text="Tecnologías",
            font=("Helvetica", 14, "bold"),
            text_color="#00c3ff"
        )
        self.tech_title.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.tech_info = ctk.CTkTextbox(
            self.tech_frame,
            fg_color="#101830",
            text_color="#e0e0e0",
            font=("Helvetica", 12),
            corner_radius=5
        )
        self.tech_info.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        
        self.tech_info.insert("1.0", """
TECNOLOGÍAS IMPLEMENTADAS

🧠 Inteligencia Artificial
- Gemini API 2.0 Flash - Modelo de lenguaje multimodal de Google
- Sistema de caché contextual con memoria neuronal
- Reconocimiento de patrones y seguimiento de gestos

👁️ Visión Artificial
- MediaPipe Hands - Detección y seguimiento de manos
- MediaPipe Face Mesh - Análisis facial de alta precisión
- OpenCV - Procesamiento avanzado de imágenes

⚙️ Desarrollo
- Python 3.11 - Lenguaje base
- CustomTkinter - Interfaz gráfica moderna
- NumPy/SciPy - Análisis numérico y procesamiento de señales
- Threading - Procesamiento asíncrono para rendimiento optimizado

REQUISITOS DEL SISTEMA
- Procesador: Intel Core i5 7ª gen. o AMD Ryzen 5 o superior
- Memoria: 8GB RAM mínimo
- Webcam: Resolución mínima 720p
- Conexión a Internet: 5 Mbps mínimo
- Sistema Operativo: Windows 10/11, macOS 12+, Ubuntu 20.04+

CARACTERÍSTICAS PRINCIPALES
- Traducción bidireccional entre LSC y texto
- Reconocimiento de gestos en tiempo real
- Conversión de texto a señas mediante GIFs
- Zoom y controles mediante gestos
- Interfaz de usuario intuitiva y moderna
        """)
        self.tech_info.configure(state="disabled")
    
    def create_settings_section(self, parent, title, row):
        """Crea una sección de título en la configuración."""
        section_frame = ctk.CTkFrame(parent, fg_color="#101840", corner_radius=5)
        section_frame.grid(row=row, column=0, padx=10, pady=(20, 10), sticky="ew")
        
        section_label = ctk.CTkLabel(
            section_frame,
            text=title,
            font=("Helvetica", 14, "bold"),
            text_color="#00c3ff"
        )
        section_label.grid(row=0, column=0, padx=10, pady=8, sticky="w")
    
    def show_frame(self, frame_name):
        """Muestra el frame seleccionado y oculta los demás."""
        # Ocultar todos los frames
        for frame in self.frames.values():
            frame.grid_remove()
        
        # Mostrar el frame seleccionado
        if frame_name in self.frames:
            self.frames[frame_name].grid(row=0, column=0, sticky="nsew")
            
            # Reproducir sonido de clic
            play_sound("click")
    
    def switch_to_signs_to_text(self):
        """Cambia al modo de señas a texto."""
        global current_mode
        current_mode = "translation"
        self.show_frame("signs_to_text")
    
    def switch_to_text_to_signs(self):
        """Cambia al modo de texto a señas."""
        global current_mode
        current_mode = "text_to_sign"
        self.show_frame("text_to_signs")
    
    def show_history(self):
        """Muestra el historial de traducciones."""
        self.show_frame("history")
        
        # Actualizar el historial
        self.update_history_display()
    
    def update_history_display(self):
        """Actualiza el contenido del historial."""
        if hasattr(self, 'history_text'):
            try:
                self.history_text.configure(state="normal")
                self.history_text.delete("1.0", tk.END)
                
                # Encabezado con estilo
                self.history_text.insert(tk.END, "🔍 HISTORIAL DE TRADUCCIONES\n")
                self.history_text.insert(tk.END, "══════════════════════════\n\n")
                
                # Mostrar traducciones con formato
                if translation_history:
                    for entry in translation_history:
                        self.history_text.insert(tk.END, f"{entry}\n")
                else:
                    self.history_text.insert(tk.END, "No hay traducciones recientes.\n")
                
                self.history_text.configure(state="disabled")
                
                # Desplazar al final para mostrar las entradas más recientes
                self.history_text.see(tk.END)
            except Exception as e:
                print(f"Error al actualizar historial: {e}")
    
    def show_settings(self):
        """Muestra la configuración."""
        self.show_frame("settings")
    
    def show_about(self):
        """Muestra información acerca del sistema."""
        self.show_frame("about")
    
    def start_camera(self):
        """Inicia la captura de video y análisis."""
        global camera_active, is_running, system_status
        
        try:
            print("🎥 Iniciando cámara...")
            camera_active = True
            is_running = True
            system_status = "analyzing"
            
            # Configurar botones
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            
            # Limpiar el texto de la etiqueta de video
            self.video_label.configure(text="")
            
            # Iniciar hilos
            video_thread = threading.Thread(target=video_stream)
            video_thread.daemon = True
            video_thread.start()
            
            api_thread = threading.Thread(target=api_worker)
            api_thread.daemon = True
            api_thread.start()
            
            # Reproducir sonido de inicio
            play_sound("scan")
            print("✅ Cámara iniciada")
        
        except Exception as e:
            print(f"Error iniciando cámara: {e}")
            try:
                messagebox.showerror("Error", f"No se pudo iniciar la cámara: {e}")
            except:
                print(f"No se pudo mostrar error: {e}")
    
    def stop_camera(self):
        """Detiene la captura de video."""
        global camera_active, system_status
        
        try:
            print("⏹️ Deteniendo cámara...")
            camera_active = False
            system_status = "standby"
            
            # Configurar botones
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            
            # Reproducir sonido de finalización
            play_sound("beep")
            print("✅ Cámara detenida")
        
        except Exception as e:
            print(f"Error deteniendo cámara: {e}")
    
    def take_screenshot(self):
        """Captura la imagen actual y la guarda."""
        try:
            # Verificar si hay imagen para capturar
            if not hasattr(self, 'video_label') or not hasattr(self.video_label, 'image'):
                try:
                    messagebox.showwarning("Captura", "No hay imagen para capturar")
                except:
                    print("No hay imagen para capturar")
                return
            
            # Crear directorio si no existe
            if not os.path.exists(CAPTURES_DIR):
                os.makedirs(CAPTURES_DIR)
            
            # Nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captura_lsc_{timestamp}.png"
            filepath = os.path.join(CAPTURES_DIR, filename)
            
            # Guardar la imagen
            img = self.video_label.image
            
            # Crear objeto Image de PIL
            pil_img = ImageTk.getimage(img)
            
            # Guardar imagen
            pil_img.save(filepath)
            
            # Mostrar confirmación
            try:
                messagebox.showinfo("Captura Guardada", f"Imagen guardada como:\n{filename}")
            except:
                print(f"Captura guardada: {filename}")
            
            # Reproducir sonido
            play_sound("success")
            
            return filepath
        
        except Exception as e:
            try:
                messagebox.showerror("Error", f"No se pudo guardar la captura: {e}")
            except:
                print(f"Error guardando captura: {e}")
            return None
    
    def export_data(self):
        """Exporta los datos de traducción."""
        try:
            # Verificar si hay traducciones para exportar
            if not translation_history:
                try:
                    messagebox.showinfo("Exportar", "No hay traducciones para exportar.")
                except:
                    print("No hay traducciones para exportar")
                return
            
            # Crear directorio si no existe
            if not os.path.exists(EXPORTS_DIR):
                os.makedirs(EXPORTS_DIR)
            
            # Nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traducciones_{timestamp}.txt"
            filepath = os.path.join(EXPORTS_DIR, filename)
            
            # Escribir archivo
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("HISTORIAL DE TRADUCCIONES LSC\n")
                f.write("============================\n\n")
                f.write(f"Fecha de exportación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for entry in translation_history:
                    f.write(f"{entry}\n")
            
            # Mostrar confirmación
            try:
                messagebox.showinfo("Exportación Completada", f"Datos exportados a:\n{filepath}")
            except:
                print(f"Datos exportados a: {filepath}")
            
            # Reproducir sonido
            play_sound("success")
            
            return filepath
        
        except Exception as e:
            try:
                messagebox.showerror("Error", f"No se pudieron exportar los datos: {e}")
            except:
                print(f"Error exportando datos: {e}")
            return None
    
    def translate_text_to_sign(self):
        """Traduce el texto ingresado a lengua de señas."""
        try:
            # Obtener el texto ingresado
            text = self.text_input.get().strip()
            
            if not text:
                try:
                    messagebox.showinfo("Traducción", "Por favor ingrese un texto para traducir.")
                except:
                    print("Por favor ingrese un texto para traducir")
                return
            
            print(f"🔍 Buscando traducción para: '{text}'")
            
            # Buscar la mejor coincidencia
            gif_file = find_best_match(text)
            
            if gif_file:
                print(f"✅ Encontrada traducción: {gif_file}")
                # Mostrar el GIF correspondiente
                self.show_sign(gif_file, text)
                
                # Añadir al historial
                timestamp = datetime.now().strftime("%H:%M:%S")
                translation_entry = f"[{timestamp}] Texto a Seña: '{text}' → {gif_file}"
                translation_history.append(translation_entry)
                
                # Mantener tamaño máximo del historial
                if len(translation_history) > max_history_size:
                    translation_history.pop(0)
                
                # Reproducir sonido de éxito
                play_sound("success")
            else:
                print(f"❌ No se encontró traducción para: '{text}'")
                # No se encontró una seña correspondiente
                self.gif_label.configure(
                    text=f"No se encontró una seña para: '{text}'\n\nPalabras disponibles: {', '.join(text_to_sign_dict.keys())}",
                    image=None
                )
                
                # Reproducir sonido de error
                play_sound("error")
        
        except Exception as e:
            print(f"Error al traducir: {e}")
            try:
                messagebox.showerror("Error", f"Error al traducir: {e}")
            except:
                print(f"Error crítico: {e}")
    
    def show_sign(self, gif_file, text=""):
        """Muestra un GIF de seña."""
        try:
            # Detener cualquier animación en curso
            if hasattr(self, 'gif_animation_id'):
                self.after_cancel(self.gif_animation_id)
            
            # Ruta completa al archivo GIF
            gif_path = os.path.join(GIFS_DIR, gif_file)
            
            # Verificar si el archivo existe
            if not os.path.exists(gif_path):
                print(f"❌ GIF no encontrado: {gif_path}")
                # Crear un mensaje indicando que falta el GIF
                self.gif_label.configure(
                    text=f"GIF no encontrado: {gif_file}\n\nColoca el archivo en: {GIFS_DIR}\n\nPalabra: {text}",
                    image=None
                )
                return
            
            print(f"📺 Mostrando GIF: {gif_file}")
            
            # Actualizar título del GIF
            if hasattr(self, 'gif_title'):
                self.gif_title.configure(
                    text=f"Traducción: {text.capitalize() if text else gif_file.replace('.gif', '').replace('_', ' ').capitalize()}"
                )
            
            # Cargar el GIF usando PIL
            self.current_gif = Image.open(gif_path)
            self.gif_frames = []
            self.current_frame_index = 0
            
            # Extraer todos los frames
            try:
                while True:
                    # Copiar el frame actual
                    frame = self.current_gif.copy()
                    
                    # Redimensionar el frame manteniendo la relación de aspecto
                    if hasattr(self, 'gif_frame'):
                        try:
                            target_width = self.gif_frame.winfo_width() - 40
                            target_height = self.gif_frame.winfo_height() - 40
                            
                            if target_width > 0 and target_height > 0:
                                # Mantener la relación de aspecto
                                img_width, img_height = frame.size
                                ratio = min(target_width/img_width, target_height/img_height)
                                new_size = (int(img_width*ratio), int(img_height*ratio))
                                
                                # Redimensionar
                                frame = frame.resize(new_size, Image.LANCZOS)
                        except:
                            # Si falla, usar tamaño por defecto
                            max_size = 400
                            img_width, img_height = frame.size
                            if img_width > max_size or img_height > max_size:
                                ratio = min(max_size/img_width, max_size/img_height)
                                new_size = (int(img_width*ratio), int(img_height*ratio))
                                frame = frame.resize(new_size, Image.LANCZOS)
                    
                    # Convertir y almacenar
                    photo = ImageTk.PhotoImage(frame)
                    self.gif_frames.append(photo)
                    
                    # Avanzar al siguiente frame
                    self.current_gif.seek(self.current_gif.tell() + 1)
            except EOFError:
                # Fin de los frames
                pass
            
            # Actualizar etiqueta y mostrar el primer frame
            if self.gif_frames:
                self.gif_label.configure(image=self.gif_frames[0], text="")
                
                # Iniciar la animación
                self.animate_gif()
                print(f"✅ GIF cargado con {len(self.gif_frames)} frames")
        
        except Exception as e:
            print(f"Error mostrando GIF: {e}")
            self.gif_label.configure(
                text=f"Error al mostrar GIF: {e}",
                image=None
            )
    
    def animate_gif(self):
        """Anima el GIF frame por frame."""
        if not self.gif_frames:
            return
        
        try:
            # Mostrar el frame actual
            frame = self.gif_frames[self.current_frame_index]
            self.gif_label.configure(image=frame)
            
            # Avanzar al siguiente frame
            self.current_frame_index = (self.current_frame_index + 1) % len(self.gif_frames)
            
            # Calcular duración para este frame (usar 100ms por defecto)
            duration = 100
            if hasattr(self.current_gif, 'info') and 'duration' in self.current_gif.info:
                duration = max(40, self.current_gif.info['duration'])  # Mínimo 40ms para evitar GIFs muy rápidos
            
            # Programar el siguiente frame
            self.gif_animation_id = self.after(duration, self.animate_gif)
        
        except Exception as e:
            print(f"Error animando GIF: {e}")
    
    def update_movement_threshold(self, value):
        """Actualiza el umbral de movimiento."""
        global movement_threshold
        movement_threshold = value
        self.movement_value.configure(text=f"{value:.3f}")
    
    def update_zoom_speed(self, value):
        """Actualiza la velocidad de zoom."""
        global zoom_speed
        zoom_speed = value
        self.zoom_value.configure(text=f"{value:.1f}")
    
    def update_api_cooldown(self, value):
        """Actualiza el tiempo entre llamadas a la API."""
        global api_cooldown
        api_cooldown = value
        self.api_value.configure(text=f"{value:.1f}")
    
    def update_ai_mode(self, value):
        """Actualiza el modo de precisión de IA."""
        global ai_precision_mode
        ai_precision_mode = value
    
    def toggle_sound(self):
        """Activa/desactiva los efectos de sonido."""
        global sound_effects_enabled
        sound_effects_enabled = bool(self.sound_var.get())
    
    def toggle_trained_gestures(self):
        """Activa/desactiva el uso de gestos entrenados."""
        global use_trained_gestures
        use_trained_gestures = bool(self.trained_gestures_var.get())
        print(f"🎯 Gestos entrenados: {'ACTIVADOS' if use_trained_gestures else 'DESACTIVADOS'}")
    
    def update_gesture_threshold(self, value):
        """Actualiza el umbral de reconocimiento de gestos."""
        global gesture_recognition_threshold
        gesture_recognition_threshold = value
        self.sensitivity_value.configure(text=f"{value:.2f}")
        print(f"🎯 Umbral de reconocimiento actualizado: {value:.2f}")
    
    def update_translation_cooldown(self, value):
        """Actualiza el tiempo de cooldown entre traducciones."""
        global translation_cooldown
        translation_cooldown = value
        self.cooldown_value.configure(text=f"{value:.1f}")
        print(f"⏱️ Cooldown de traducciones actualizado: {value:.1f}s")
    
    def reload_trained_gestures(self):
        """Recarga los gestos entrenados."""
        print("🔄 Recargando gestos entrenados...")
        load_trained_gestures()
        try:
            messagebox.showinfo("Gestos", "Gestos entrenados recargados correctamente")
        except:
            print("Gestos entrenados recargados correctamente")
    
    def save_config(self):
        """Guarda la configuración actual."""
        config_path = os.path.join(DATA_DIR, "config.json")
        try:
            config = {
                "zoom_speed": zoom_speed,
                "movement_threshold": movement_threshold,
                "api_cooldown": api_cooldown,
                "sound_effects_enabled": sound_effects_enabled,
                "ai_precision_mode": ai_precision_mode,
                "camera_index": camera_index,
                "camera_resolution": camera_resolution,
                "frame_rate": frame_rate,
                "synonyms": synonym_dict,
                "use_trained_gestures": use_trained_gestures,
                "gesture_recognition_threshold": gesture_recognition_threshold,
                "translation_cooldown": translation_cooldown,
                "processing_frame_skip": processing_frame_skip,
                "gesture_preparation_time": gesture_preparation_time,
                "gesture_ready_threshold": gesture_ready_threshold,
                "hand_stability_threshold": hand_stability_threshold
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            try:
                messagebox.showinfo("Configuración", "Configuración guardada correctamente")
            except:
                print("Configuración guardada correctamente")
        except Exception as e:
            try:
                messagebox.showerror("Error", f"No se pudo guardar la configuración: {e}")
            except:
                print(f"Error guardando configuración: {e}")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación."""
        global is_running, camera_active
        
        try:
            confirm = True
            try:
                confirm = messagebox.askokcancel("Salir", "¿Está seguro de que desea salir?")
            except:
                print("Cerrando aplicación...")
            
            if confirm:
                print("🚪 Cerrando aplicación...")
                
                # Detener captura de video y procesamiento
                camera_active = False
                is_running = False
                
                # Esperar un momento para que los hilos se detengan
                time.sleep(0.5)
                
                # Cerrar la aplicación
                self.quit()
                self.destroy()
                print("✅ Aplicación cerrada")
        except Exception as e:
            print(f"Error cerrando aplicación: {e}")

    def create_training_frame(self):
        """Crea el frame para entrenamiento de gestos."""
        # Frame de entrenamiento
        self.frames["training"] = ctk.CTkFrame(self.content_frame, fg_color="#15202b")
        self.frames["training"].grid(row=0, column=0, sticky="nsew")
        self.frames["training"].grid_rowconfigure(0, weight=0)  # Título
        self.frames["training"].grid_rowconfigure(1, weight=0)  # Entrada
        self.frames["training"].grid_rowconfigure(2, weight=1)  # Video
        self.frames["training"].grid_rowconfigure(3, weight=0)  # Controles
        self.frames["training"].grid_columnconfigure(0, weight=1)
        
        # Título
        self.training_title_frame = ctk.CTkFrame(self.frames["training"], fg_color="#101825", corner_radius=10)
        self.training_title_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.training_title = ctk.CTkLabel(
            self.training_title_frame, 
            text="🎯 ENTRENAMIENTO DE GESTOS",
            font=("Helvetica", 18, "bold"),
            text_color="#00c3ff"
        )
        self.training_title.grid(row=0, column=0, padx=20, pady=10)
        
        # Subtítulo con información
        self.training_subtitle = ctk.CTkLabel(
            self.training_title_frame,
            text="Entrena nuevos gestos realizando cada movimiento varias veces\nUsa el botón 'Capturar Gesto' o cierra los ojos por 1 segundo para capturar",
            font=("Helvetica", 11),
            text_color="#ffffff"
        )
        self.training_subtitle.grid(row=1, column=0, padx=10, pady=(0, 10))
        
        # Marco para entrada del nombre del gesto
        self.gesture_input_frame = ctk.CTkFrame(
            self.frames["training"],
            fg_color="#0a1020",
            corner_radius=10
        )
        self.gesture_input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.gesture_input_frame.grid_columnconfigure(0, weight=1)
        self.gesture_input_frame.grid_columnconfigure(1, weight=0)
        
        # Campo de entrada para el nombre del gesto
        self.gesture_input = ctk.CTkEntry(
            self.gesture_input_frame,
            placeholder_text="Nombre del gesto a entrenar...",
            font=("Helvetica", 14),
            height=40,
            fg_color="#101830",
            text_color="#ffffff",
            border_color="#00a0e0",
            border_width=2
        )
        self.gesture_input.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Botón de iniciar entrenamiento
        self.start_training_btn = ctk.CTkButton(
            self.gesture_input_frame,
            text="Iniciar Entrenamiento",
            command=self.start_gesture_training,
            font=("Helvetica", 13, "bold"),
            fg_color="#004080",
            hover_color="#005fa0",
            width=150,
            height=40,
            corner_radius=8
        )
        self.start_training_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Marco para el video
        self.training_video_frame = ctk.CTkFrame(
            self.frames["training"], 
            fg_color="#0a1020",
            corner_radius=15,
            border_width=2,
            border_color="#00a0e0"
        )
        self.training_video_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.training_video_frame.grid_propagate(False)
        
        # Etiqueta para mostrar el video
        self.training_video_label = ctk.CTkLabel(
            self.training_video_frame, 
            text="Ingresa el nombre del gesto y presiona 'Iniciar Entrenamiento'\nPodrás capturar con el botón o parpadeando 1 segundo",
            font=("Helvetica", 16)
        )
        self.training_video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Panel de controles de entrenamiento
        self.training_controls_frame = ctk.CTkFrame(
            self.frames["training"], 
            fg_color="#101825",
            corner_radius=10
        )
        self.training_controls_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        self.training_controls_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Botones de control
        self.btn_capture_gesture = ctk.CTkButton(
            self.training_controls_frame, 
            text="📸 CAPTURAR GESTO\n(o parpadea 1s)",
            command=self.capture_gesture,
            font=("Helvetica", 12, "bold"),
            fg_color="#006060",
            hover_color="#008080",
            height=50,
            corner_radius=8,
            state="disabled"
        )
        self.btn_capture_gesture.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_finish_training = ctk.CTkButton(
            self.training_controls_frame, 
            text="✅ FINALIZAR ENTRENAMIENTO",
            command=self.finish_training,
            font=("Helvetica", 13, "bold"),
            fg_color="#004080",
            hover_color="#005fa0",
            height=45,
            corner_radius=8,
            state="disabled"
        )
        self.btn_finish_training.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_cancel_training = ctk.CTkButton(
            self.training_controls_frame, 
            text="❌ CANCELAR",
            command=self.cancel_training,
            font=("Helvetica", 13, "bold"),
            fg_color="#802000",
            hover_color="#a03000",
            height=45,
            corner_radius=8,
            state="disabled"
        )
        self.btn_cancel_training.grid(row=0, column=2, padx=10, pady=10)
        
        # Barra de progreso
        self.training_progress = ctk.CTkProgressBar(
            self.training_controls_frame,
            mode="determinate",
            height=20,
            corner_radius=10
        )
        self.training_progress.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self.training_progress.set(0)
        
        # Etiqueta de progreso
        self.training_progress_label = ctk.CTkLabel(
            self.training_controls_frame,
            text=f"Muestras: 0/{TRAINING_MIN_SAMPLES}",
            font=("Helvetica", 12),
            text_color="#00c3ff"
        )
        self.training_progress_label.grid(row=2, column=0, columnspan=3, pady=5)
    
    def switch_to_training(self):
        """Cambia al modo de entrenamiento."""
        global current_mode
        current_mode = "training"
        self.show_frame("training")
    
    def start_gesture_training(self):
        """Inicia una nueva sesión de entrenamiento."""
        global training_mode, current_gesture_name, training_samples
        global training_counter, training_in_progress, training_ready
        
        # Obtener el nombre del gesto
        gesture_name = self.gesture_input.get().strip()
        
        if not gesture_name:
            try:
                messagebox.showwarning("Entrenamiento", "Por favor ingrese un nombre para el gesto")
            except:
                print("Por favor ingrese un nombre para el gesto")
            return
        
        # Normalizar el nombre del gesto
        gesture_name = normalize_text(gesture_name)
        
        # Verificar si ya existe
        if gesture_name in text_to_sign_dict:
            try:
                messagebox.showwarning("Entrenamiento", f"El gesto '{gesture_name}' ya existe")
            except:
                print(f"El gesto '{gesture_name}' ya existe")
            return
        
        # Inicializar variables de entrenamiento
        training_mode = True
        current_gesture_name = gesture_name
        training_samples = []
        training_counter = 0
        training_in_progress = True
        training_ready = False
        
        # Actualizar interfaz
        self.gesture_input.configure(state="disabled")
        self.start_training_btn.configure(state="disabled")
        self.btn_capture_gesture.configure(state="normal")
        self.btn_finish_training.configure(state="disabled")
        self.btn_cancel_training.configure(state="normal")
        
        # Iniciar cámara si no está activa
        if not camera_active:
            self.start_camera()
        
        # Actualizar etiqueta de video
        self.training_video_label.configure(
            text=f"Realiza el gesto '{gesture_name}'\nPresiona 'Capturar Gesto' o parpadea 1 segundo para capturar"
        )
        
        # Reproducir sonido
        play_sound("scan")
    
    def capture_gesture(self):
        """Captura una muestra del gesto actual."""
        global training_counter, training_ready, training_samples
        
        try:
            if not training_in_progress:
                return
            
            # Verificar si tenemos frame actual
            if not hasattr(self.training_video_label, 'image'):
                try:
                    messagebox.showwarning("Entrenamiento", "Espere a que la cámara esté activa")
                except:
                    print("Espere a que la cámara esté activa")
                return
            
            # Capturar datos del gesto actual
            if resultados_manos and resultados_manos.multi_hand_landmarks:
                # Extraer características del gesto
                gesture_data = []
                for hand_landmarks in resultados_manos.multi_hand_landmarks:
                    # Convertir landmarks a formato numpy para procesamiento
                    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    gesture_data.append(landmarks_array)
                
                # Guardar muestra
                training_samples.append(gesture_data)
                training_counter += 1
                
                # Actualizar progreso
                progress = min(1.0, training_counter / TRAINING_MIN_SAMPLES)
                self.training_progress.set(progress)
                self.training_progress_label.configure(
                    text=f"Muestras: {training_counter}/{TRAINING_MIN_SAMPLES}"
                )
                
                # Verificar si tenemos suficientes muestras
                if training_counter >= TRAINING_MIN_SAMPLES:
                    training_ready = True
                    self.btn_finish_training.configure(state="normal")
                
                # Reproducir sonido
                play_sound("capture")
            else:
                try:
                    messagebox.showwarning("Entrenamiento", "No se detectaron manos en la imagen")
                except:
                    print("No se detectaron manos en la imagen")
        
        except Exception as e:
            print(f"Error capturando gesto: {e}")
            try:
                messagebox.showerror("Error", f"Error capturando gesto: {e}")
            except:
                print(f"Error crítico: {e}")
    
    def finish_training(self):
        """Finaliza el entrenamiento y guarda el modelo."""
        global training_mode, training_in_progress, training_samples
        global training_counter, training_ready, current_gesture_name
        
        try:
            if not training_ready or not training_samples:
                return
            
            # Crear directorio si no existe
            if not os.path.exists(GESTURES_DIR):
                os.makedirs(GESTURES_DIR)
            
            # Convertir arrays de NumPy a listas
            processed_samples = []
            for sample in training_samples:
                processed_sample = []
                for hand_data in sample:
                    # Convertir el array de NumPy a lista
                    hand_data_list = hand_data.tolist()
                    processed_sample.append(hand_data_list)
                processed_samples.append(processed_sample)
            
            # Preparar datos para guardar
            gesture_data = {
                "name": current_gesture_name,
                "samples": processed_samples,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(training_samples)
            }
            
            # Guardar datos de entrenamiento
            gesture_file = os.path.join(GESTURES_DIR, f"{current_gesture_name}.json")
            with open(gesture_file, "w") as f:
                json.dump(gesture_data, f, indent=2)
            
            # Actualizar diccionario de gestos
            text_to_sign_dict[current_gesture_name] = f"{current_gesture_name}.gif"
            
            # Recargar gestos entrenados para incluir el nuevo
            print("🔄 Recargando gestos entrenados...")
            load_trained_gestures()
            
            # Mostrar mensaje de éxito
            try:
                messagebox.showinfo("Entrenamiento", 
                                  f"Entrenamiento completado con éxito.\n"
                                  f"Se guardaron {len(training_samples)} muestras del gesto '{current_gesture_name}'")
            except:
                print(f"Entrenamiento completado: {len(training_samples)} muestras guardadas")
            
            # Reproducir sonido
            play_sound("success")
            
            # Reiniciar todas las variables globales
            training_mode = False
            training_in_progress = False
            training_samples = []
            training_counter = 0
            training_ready = False
            current_gesture_name = ""
            
            # Reiniciar la interfaz
            self.gesture_input.configure(state="normal")
            self.gesture_input.delete(0, tk.END)
            self.start_training_btn.configure(state="normal")
            self.btn_capture_gesture.configure(state="disabled")
            self.btn_finish_training.configure(state="disabled")
            self.btn_cancel_training.configure(state="disabled")
            
            # Reiniciar la barra de progreso
            self.training_progress.set(0)
            self.training_progress_label.configure(text=f"Muestras: 0/{TRAINING_MIN_SAMPLES}")
            
            # Reiniciar el texto de la etiqueta de video
            self.training_video_label.configure(
                text="Ingresa el nombre del gesto y presiona 'Iniciar Entrenamiento'\nPodrás capturar con el botón o parpadeando 1 segundo"
            )
            
            # Detener cámara
            self.stop_camera()
        
        except Exception as e:
            print(f"Error finalizando entrenamiento: {e}")
            try:
                messagebox.showerror("Error", f"Error finalizando entrenamiento: {e}")
            except:
                print(f"Error crítico: {e}")
    
    def cancel_training(self):
        """Cancela la sesión de entrenamiento actual."""
        global training_mode, training_in_progress
        
        try:
            # Reiniciar variables
            training_mode = False
            training_in_progress = False
            
            # Actualizar interfaz
            self.gesture_input.configure(state="normal")
            self.gesture_input.delete(0, tk.END)
            self.start_training_btn.configure(state="normal")
            self.btn_capture_gesture.configure(state="disabled")
            self.btn_finish_training.configure(state="disabled")
            self.btn_cancel_training.configure(state="disabled")
            
            # Reiniciar progreso
            self.training_progress.set(0)
            self.training_progress_label.configure(text=f"Muestras: 0/{TRAINING_MIN_SAMPLES}")
            
            # Detener cámara
            self.stop_camera()
            
            # Reproducir sonido
            play_sound("beep")
        
        except Exception as e:
            print(f"Error cancelando entrenamiento: {e}")

# Función para comprobar la existencia de GIFs y crear estructura si no existe
def check_directories():
    """Verifica y crea los directorios necesarios."""
    for directory in [DATA_DIR, CACHE_DIR, GESTURES_DIR, CAPTURES_DIR, EXPORTS_DIR, SOUNDS_DIR, GIFS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directorio creado: {directory}")

# Función para cargar la configuración guardada
def load_config():
    """Carga la configuración desde el archivo."""
    global zoom_speed, movement_threshold, api_cooldown, sound_effects_enabled, ai_precision_mode
    global camera_index, camera_resolution, frame_rate, synonym_dict
    global use_trained_gestures, gesture_recognition_threshold
    global translation_cooldown, processing_frame_skip
    global gesture_preparation_time, gesture_ready_threshold, hand_stability_threshold
    
    config_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Cargar parámetros
            zoom_speed = config.get("zoom_speed", zoom_speed)
            movement_threshold = config.get("movement_threshold", movement_threshold)
            api_cooldown = config.get("api_cooldown", api_cooldown)
            sound_effects_enabled = config.get("sound_effects_enabled", sound_effects_enabled)
            ai_precision_mode = config.get("ai_precision_mode", ai_precision_mode)
            camera_index = config.get("camera_index", camera_index)
            camera_resolution = config.get("camera_resolution", camera_resolution)
            frame_rate = config.get("frame_rate", frame_rate)
            
            # Cargar configuraciones de gestos entrenados
            use_trained_gestures = config.get("use_trained_gestures", use_trained_gestures)
            gesture_recognition_threshold = config.get("gesture_recognition_threshold", gesture_recognition_threshold)
            
            # Cargar configuraciones de estabilización mejoradas
            translation_cooldown = config.get("translation_cooldown", 1.0)  # Nuevo valor por defecto más rápido
            processing_frame_skip = config.get("processing_frame_skip", 3)  # Nuevo valor por defecto más frecuente
            gesture_preparation_time = config.get("gesture_preparation_time", 0.3)  # Muy rápido
            gesture_ready_threshold = config.get("gesture_ready_threshold", 0.5)  # Rápido
            hand_stability_threshold = config.get("hand_stability_threshold", 0.08)  # Más tolerante
            
            # Cargar sinónimos si existen
            if "synonyms" in config:
                synonym_dict.update(config["synonyms"])
            
            print("✅ Configuración cargada correctamente")
            print(f"   📊 Cooldown de traducción: {translation_cooldown}s")
            print(f"   📊 Tiempo de preparación: {gesture_preparation_time}s")
            print(f"   📊 Frame skip: cada {processing_frame_skip} frames")
        except Exception as e:
            print(f"⚠️  Error al cargar la configuración: {e}")
    else:
        print("📝 Usando configuración por defecto optimizada para estabilidad")

# Función para iniciar la aplicación
def main():
    """Función principal para iniciar la aplicación."""
    global app, is_running
    
    try:
        print("=" * 60)
        print("🚀 INICIANDO TRADUCTOR LSC BIDIRECCIONAL")
        print("=" * 60)
        
        # Verificar directorio actual
        print(f"📁 Directorio actual: {os.getcwd()}")
        
        # Inicializar variables globales
        is_running = True
        
        # Verificar y crear directorios
        check_directories()
        
        # Cargar configuración
        load_config()
        
        # Cargar gestos entrenados
        print("🧠 Cargando gestos entrenados...")
        load_trained_gestures()
        
        print("🎨 Creando interfaz de usuario...")
        
        # Crear y ejecutar la aplicación
        app = SignLanguageTranslatorApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("✅ Aplicación lista - Iniciando bucle principal")
        print("=" * 60)
        
        # Iniciar el bucle principal
        app.mainloop()
        
        print("👋 Aplicación finalizada")
        is_running = False
    
    except Exception as e:
        print(f"❌ Error crítico al iniciar la aplicación: {e}")
        
        try:
            messagebox.showerror("Error Fatal", f"Error al iniciar:\n\n{e}\n\nVerifica que todas las dependencias estén instaladas.")
        except:
            print("No se pudo mostrar el diálogo de error")
        
        is_running = False
        
        # Esperar entrada del usuario antes de cerrar
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("data"):
        print("📁 Creando estructura de directorios...")
        create_directories()
    
    # Ejecutar aplicación principal
    main()