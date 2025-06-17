import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import threading
import time
from typing import List, Dict, Tuple, Optional
import logging
import os
from dataclasses import dataclass
from collections import deque
from database.gesture_db import GestureDatabase
from models.gesture_model import GestureModel

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GestureFrame:
    landmarks: np.ndarray
    timestamp: float
    confidence: float

class GestureBuffer:
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: Dict):
        with self.lock:
            self.buffer.append(frame)
    
    def get_sequence(self) -> List[Dict]:
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        with self.lock:
            self.buffer.clear()

class LSCGestureRecognizer:
    def __init__(self, model_path: str, db_path: str = "data/gestures.db"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Cargar base de datos
        self.db = GestureDatabase(db_path)
        
        # Cargar modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
        # Buffer para secuencias de gestos
        self.gesture_buffer = GestureBuffer()
        
        # Estado de reconocimiento
        self.current_gesture = None
        self.gesture_start_time = None
        self.min_gesture_duration = 0.5  # segundos
        self.confidence_threshold = 0.85
    
    def _load_model(self, model_path: str) -> GestureModel:
        """Cargar modelo entrenado"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
        # Obtener número de clases de la base de datos
        num_classes = len(self.db.get_all_gestures())
        
        # Crear y cargar modelo
        model = GestureModel(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Procesar un frame de video"""
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Obtener landmarks
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Dibujar landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Convertir a tensor
            landmarks = np.array(landmarks)
            landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Añadir al buffer
            self.gesture_buffer.add_frame({
                "landmarks": landmarks,
                "timestamp": time.time()
            })
            
            # Reconocer gesto si el buffer está lleno
            if len(self.gesture_buffer.get_sequence()) == self.gesture_buffer.maxlen:
                return self._recognize_gesture()
        
        return None
    
    def _recognize_gesture(self) -> Optional[Dict]:
        """Reconocer gesto a partir de la secuencia en el buffer"""
        sequence = self.gesture_buffer.get_sequence()
        
        # Convertir secuencia a tensor
        landmarks = np.array([frame["landmarks"] for frame in sequence])
        landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
        
        # Realizar predicción
        with torch.no_grad():
            logits, attention_weights = self.model(landmarks_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            
            if confidence.item() > self.confidence_threshold:
                # Obtener información del gesto
                gesture_id = pred.item()
                gesture = self.db.get_gesture(gesture_id)
                
                if gesture:
                    # Verificar duración mínima
                    current_time = time.time()
                    if self.current_gesture != gesture["name"]:
                        self.current_gesture = gesture["name"]
                        self.gesture_start_time = current_time
                    elif current_time - self.gesture_start_time >= self.min_gesture_duration:
                        return {
                            "gesture": gesture["name"],
                            "confidence": confidence.item(),
                            "attention_weights": attention_weights.squeeze().cpu().numpy()
                        }
        
        return None

class LSCTranslator:
    def __init__(self, model_path: str, db_path: str = "data/gestures.db"):
        self.gesture_recognizer = LSCGestureRecognizer(model_path, db_path)
        self.db = GestureDatabase(db_path)
        self.current_phrase = []
        self.phrase_buffer = []
        self.confidence_threshold = 0.85
    
    def process_video(self, frame: np.ndarray) -> Optional[str]:
        """Procesar frame de video y traducir gestos"""
        result = self.gesture_recognizer.process_frame(frame)
        
        if result and result["confidence"] > self.confidence_threshold:
            # Obtener traducciones del gesto
            gesture = self.db.search_gesture_by_name(result["gesture"])[0]
            translations = self.db.get_gesture_translations(gesture["id"])
            
            if translations:
                self.phrase_buffer.append(translations[0])
                
                # Verificar si tenemos una frase completa
                if self._is_complete_phrase():
                    phrase = " ".join(self.phrase_buffer)
                    self.phrase_buffer = []
                    return phrase
        
        return None
    
    def _is_complete_phrase(self) -> bool:
        """Determinar si tenemos una frase completa"""
        # TODO: Implementar lógica más sofisticada para determinar frases completas
        return len(self.phrase_buffer) >= 3

def main():
    # Configurar rutas
    model_path = "models/gesture_model.pth"
    db_path = "data/gestures.db"
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        logger.error(f"No se encontró el modelo en {model_path}")
        logger.info("Por favor, entrena el modelo primero con train_model.py")
        return
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara")
        return
    
    # Inicializar traductor
    translator = LSCTranslator(model_path, db_path)
    
    logger.info("Iniciando sistema de traducción LSC...")
    logger.info("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error al leer frame de la cámara")
            break
        
        # Procesar frame
        result = translator.process_video(frame)
        
        # Mostrar resultado
        if result:
            logger.info(f"Frase detectada: {result}")
            # Dibujar texto en el frame
            cv2.putText(
                frame,
                result,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Mostrar frame
        cv2.imshow("LSC Translator", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()