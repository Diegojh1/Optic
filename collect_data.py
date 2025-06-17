import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import logging
import argparse
from typing import List, Dict, Optional
from database.gesture_db import GestureDatabase
import sqlite3

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, db_path: str = "data/gestures.db"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar base de datos
        self.db = GestureDatabase(db_path)
        
        # Estado de grabación
        self.is_recording = False
        self.current_sequence = []
        self.current_gesture_id = None
        self.frame_count = 0
        self.min_frames = 30
        
        # Colores para feedback visual
        self.COLORS = {
            'recording': (0, 0, 255),  # Rojo
            'ready': (0, 255, 0),      # Verde
            'text': (255, 255, 255)    # Blanco
        }
    
    def start_recording(self, gesture_name: str, description: str = "", category: str = "general") -> bool:
        """Iniciar la grabación de un nuevo gesto"""
        if self.is_recording:
            logger.warning("Ya hay una grabación en progreso")
            return False
        
        # Buscar si el gesto ya existe
        existing_gestures = self.db.search_gesture_by_name(gesture_name)
        if existing_gestures:
            self.current_gesture_id = existing_gestures[0]["id"]
            logger.info(f"Continuando grabación para el gesto existente: {gesture_name}")
        else:
            # Crear nuevo gesto
            self.current_gesture_id = self.db.add_gesture(gesture_name, description, category)
            logger.info(f"Iniciando grabación para nuevo gesto: {gesture_name}")
        
        self.is_recording = True
        self.current_sequence = []
        self.frame_count = 0
        return True
    
    def stop_recording(self) -> bool:
        """Detener la grabación y guardar la secuencia"""
        if not self.is_recording:
            logger.warning("No hay grabación en progreso")
            return False
        
        if len(self.current_sequence) < self.min_frames:
            logger.warning(f"Secuencia demasiado corta ({len(self.current_sequence)} frames). Mínimo: {self.min_frames}")
            self.is_recording = False
            self.current_sequence = []
            return False
        
        # Convertir secuencia a numpy array
        sequence = np.array(self.current_sequence)
        
        # Calcular confianza promedio
        confidence = np.mean([frame["confidence"] for frame in self.current_sequence])
        
        # Guardar en la base de datos
        sequence_id = self.db.add_sequence(self.current_gesture_id, sequence, confidence)
        
        # Obtener información del gesto
        gesture = self.db.get_gesture(self.current_gesture_id)
        sequences = self.db.get_gesture_sequences(self.current_gesture_id)
        
        logger.info(f"Secuencia guardada para el gesto '{gesture['name']}' (ID: {sequence_id})")
        logger.info(f"Total de secuencias para este gesto: {len(sequences)}")
        
        self.is_recording = False
        self.current_sequence = []
        return True
    
    def delete_last_sequence(self) -> bool:
        """Eliminar la última secuencia grabada"""
        if not self.current_gesture_id:
            logger.warning("No hay gesto seleccionado")
            return False
        
        sequences = self.db.get_gesture_sequences(self.current_gesture_id)
        if not sequences:
            logger.warning("No hay secuencias para eliminar")
            return False
        
        # Obtener el ID de la última secuencia
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM sequences WHERE gesture_id = ? ORDER BY id DESC LIMIT 1",
                (self.current_gesture_id,)
            )
            result = cursor.fetchone()
            if result:
                last_sequence_id = result[0]
                # Eliminar la secuencia
                if self.db.delete_sequence(last_sequence_id):
                    logger.info(f"Secuencia {last_sequence_id} eliminada")
                    return True
        
        return False
    
    def _draw_status(self, frame: np.ndarray):
        """Dibujar información de estado en el frame"""
        h, w = frame.shape[:2]
        
        # Dibujar estado de grabación
        status = "GRABANDO" if self.is_recording else "LISTO"
        color = self.COLORS['recording'] if self.is_recording else self.COLORS['ready']
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Dibujar contador de frames
        if self.is_recording:
            cv2.putText(
                frame,
                f"Frames: {len(self.current_sequence)}/{self.min_frames}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.COLORS['text'],
                2
            )
        
        # Dibujar instrucciones
        instructions = [
            "Presiona 'r' para iniciar/detener grabación",
            "Presiona 'd' para eliminar última secuencia",
            "Presiona 'q' para salir"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(
                frame,
                text,
                (10, h - 30 - (i * 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.COLORS['text'],
                2
            )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesar un frame de video"""
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        results = self.hands.process(frame_rgb)
        
        # Dibujar landmarks si se detectan manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Si estamos grabando, guardar landmarks
                if self.is_recording:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    confidence = results.multi_handedness[0].classification[0].score
                    self.current_sequence.append({
                        "landmarks": landmarks,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })
        
        # Dibujar estado
        self._draw_status(frame)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description="Recolector de datos para LSC")
    parser.add_argument("--db_path", default="data/gestures.db", help="Ruta a la base de datos")
    args = parser.parse_args()
    
    # Crear directorio de datos si no existe
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara")
        return
    
    collector = DataCollector(args.db_path)
    
    logger.info("Iniciando recolección de datos...")
    logger.info("Presiona 'r' para iniciar/detener la grabación de un gesto")
    logger.info("Presiona 'd' para eliminar la última secuencia grabada")
    logger.info("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error al leer frame de la cámara")
            break
        
        # Procesar frame
        frame = collector.process_frame(frame)
        
        # Mostrar frame
        cv2.imshow("LSC Data Collector", frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not collector.is_recording:
                # Iniciar grabación
                gesture_name = input("Nombre del gesto: ")
                description = input("Descripción (opcional): ")
                category = input("Categoría (opcional, default: general): ") or "general"
                collector.start_recording(gesture_name, description, category)
            else:
                # Detener grabación
                collector.stop_recording()
        elif key == ord('d'):
            collector.delete_last_sequence()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 