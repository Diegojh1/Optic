import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureDatabase:
    def __init__(self, db_path: str = "data/gestures.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializar la base de datos con las tablas necesarias"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de gestos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de secuencias
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_id INTEGER,
                    landmarks BLOB,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gesture_id) REFERENCES gestures(id)
                )
            """)
            
            # Tabla de traducciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_id INTEGER,
                    text TEXT NOT NULL,
                    language TEXT DEFAULT 'es',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gesture_id) REFERENCES gestures(id)
                )
            """)
            
            conn.commit()
    
    def add_gesture(self, name: str, description: str = "", category: str = "general") -> int:
        """Agregar un nuevo gesto a la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO gestures (name, description, category) VALUES (?, ?, ?)",
                (name, description, category)
            )
            conn.commit()
            return cursor.lastrowid
    
    def add_sequence(self, gesture_id: int, landmarks: np.ndarray, confidence: float) -> int:
        """Agregar una nueva secuencia de gesto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sequences (gesture_id, landmarks, confidence) VALUES (?, ?, ?)",
                (gesture_id, landmarks.tobytes(), confidence)
            )
            conn.commit()
            return cursor.lastrowid
    
    def add_translation(self, gesture_id: int, text: str, language: str = "es") -> int:
        """Agregar una traducción para un gesto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO translations (gesture_id, text, language) VALUES (?, ?, ?)",
                (gesture_id, text, language)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_gesture(self, gesture_id: int) -> Optional[Dict]:
        """Obtener información de un gesto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures WHERE id = ?", (gesture_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                }
            return None
    
    def get_gesture_sequences(self, gesture_id: int) -> List[Tuple[np.ndarray, float]]:
        """Obtener todas las secuencias de un gesto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT landmarks, confidence FROM sequences WHERE gesture_id = ?", (gesture_id,))
            sequences = []
            for row in cursor.fetchall():
                landmarks = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 3)
                sequences.append((landmarks, row[1]))
            return sequences
    
    def get_gesture_translations(self, gesture_id: int, language: str = "es") -> List[str]:
        """Obtener todas las traducciones de un gesto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM translations WHERE gesture_id = ? AND language = ?",
                (gesture_id, language)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def search_gesture_by_name(self, name: str) -> List[Dict]:
        """Buscar gestos por nombre"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures WHERE name LIKE ?", (f"%{name}%",))
            return [{
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "category": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            } for row in cursor.fetchall()]
    
    def get_all_gestures(self) -> List[Dict]:
        """Obtener todos los gestos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures")
            return [{
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "category": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            } for row in cursor.fetchall()]
    
    def delete_gesture(self, gesture_id: int) -> bool:
        """Eliminar un gesto y todas sus secuencias y traducciones"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sequences WHERE gesture_id = ?", (gesture_id,))
                cursor.execute("DELETE FROM translations WHERE gesture_id = ?", (gesture_id,))
                cursor.execute("DELETE FROM gestures WHERE id = ?", (gesture_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error al eliminar gesto: {e}")
            return False
    
    def delete_sequence(self, sequence_id: int) -> bool:
        """Eliminar una secuencia específica"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sequences WHERE id = ?", (sequence_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error al eliminar secuencia: {e}")
            return False 