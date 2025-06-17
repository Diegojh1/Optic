import os
import requests
import zipfile
import cv2
import numpy as np
import json
import logging
from tqdm import tqdm
from database.gesture_db import GestureDatabase
import mediapipe as mp
from typing import List, Dict
import pandas as pd

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSC50Downloader:
    def __init__(self, download_dir: str = "data/lsc50"):
        self.download_dir = download_dir
        self.dataset_url = "https://figshare.com/ndownloader/files/50061846"  # URL del dataset LSC50
        self.db = GestureDatabase("data/gestures.db")
        
        # Lista de 50 señas del dataset LSC50
        self.lsc50_signs = [
            "ABUELO", "ABUELA", "AGUA", "AMARILLO", "AZUL", "BLANCO", "BUENO", "CASA",
            "COMER", "COMO", "CUANDO", "DONDE", "DORMIR", "FELIZ", "GATO", "GRACIAS",
            "GRANDE", "HERMANA", "HERMANO", "HOLA", "MALO", "MAMA", "NEGRO", "PAPA",
            "PEQUEÑO", "PERRO", "POR FAVOR", "QUE", "QUIEN", "ROJO", "TRISTE", "USTED",
            "VERDE", "YO", "BAÑAR", "CALIENTE", "CAMA", "CAMINAR", "CARRO", "COCINA",
            "DULCE", "ESCUELA", "FRIO", "LECHE", "LLORAR", "MEDICINA", "PAIN", "REIR",
            "SOL", "TRABAJO"
        ]
        
        # MediaPipe para extraer landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def download_dataset(self) -> bool:
        """Descargar el dataset LSC50"""
        try:
            os.makedirs(self.download_dir, exist_ok=True)
            
            # Archivo de descarga
            zip_path = os.path.join(self.download_dir, "lsc50_dataset.zip")
            
            if os.path.exists(zip_path):
                logger.info("El dataset ya está descargado")
                return True
            
            logger.info("Descargando dataset LSC50...")
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as file, tqdm(
                desc="Descargando",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    pbar.update(size)
            
            logger.info("Descarga completada")
            return True
            
        except Exception as e:
            logger.error(f"Error al descargar el dataset: {e}")
            return False
    
    def extract_dataset(self) -> bool:
        """Extraer el dataset descargado"""
        try:
            zip_path = os.path.join(self.download_dir, "lsc50_dataset.zip")
            extract_path = os.path.join(self.download_dir, "extracted")
            
            if os.path.exists(extract_path):
                logger.info("El dataset ya está extraído")
                return True
            
            logger.info("Extrayendo dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            logger.info("Extracción completada")
            return True
            
        except Exception as e:
            logger.error(f"Error al extraer el dataset: {e}")
            return False
    
    def process_video(self, video_path: str) -> List[np.ndarray]:
        """Procesar un video y extraer landmarks"""
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar manos
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Extraer landmarks de ambas manos
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Asegurar que tenemos exactamente 42 landmarks (2 manos x 21 landmarks)
                while len(landmarks) < 42:
                    landmarks.append([0.0, 0.0, 0.0])  # Padding para manos faltantes
                landmarks = landmarks[:42]  # Truncar si hay más de 2 manos
                
                landmarks_sequence.append(np.array(landmarks))
        
        cap.release()
        return landmarks_sequence
    
    def import_to_database(self) -> bool:
        """Importar datos procesados a la base de datos"""
        try:
            extract_path = os.path.join(self.download_dir, "extracted")
            video_path = os.path.join(extract_path, "Videos", "RGB_Body")
            
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el directorio de videos: {video_path}")
                return False
            
            logger.info("Importando datos a la base de datos...")
            
            for i, sign_name in enumerate(self.lsc50_signs):
                # Crear gesto en la base de datos
                gesture_id = self.db.add_gesture(
                    name=sign_name,
                    description=f"Seña LSC del dataset LSC50",
                    category="lsc50"
                )
                
                # Agregar traducción
                self.db.add_translation(gesture_id, sign_name.lower(), "es")
                
                # Buscar videos para esta seña
                sign_number = f"{i+1:04d}"  # Formato 0001, 0002, etc.
                
                video_files = [f for f in os.listdir(video_path) if f.startswith(sign_number)]
                
                for video_file in video_files:
                    video_full_path = os.path.join(video_path, video_file)
                    
                    # Procesar video
                    landmarks_sequence = self.process_video(video_full_path)
                    
                    if landmarks_sequence:
                        # Convertir secuencia a array numpy
                        sequence_array = np.array(landmarks_sequence)
                        
                        # Agregar secuencia a la base de datos
                        self.db.add_sequence(gesture_id, sequence_array, 0.9)
                        logger.info(f"Procesado video {video_file} para seña {sign_name}")
            
            logger.info("Importación completada")
            return True
            
        except Exception as e:
            logger.error(f"Error al importar datos: {e}")
            return False
    
    def create_gesture_translations(self):
        """Crear traducciones adicionales para los gestos"""
        translations_dict = {
            "HOLA": ["hola", "saludo", "saludar"],
            "GRACIAS": ["gracias", "agradecer", "agradecimiento"],
            "BUENOS DIAS": ["buenos días", "buen día", "saludo matutino"],
            "CASA": ["casa", "hogar", "vivienda", "domicilio"],
            "FAMILIA": ["familia", "parientes", "familiares"],
            "AGUA": ["agua", "líquido", "beber agua"],
            "COMER": ["comer", "alimentarse", "comida"],
            "DORMIR": ["dormir", "descansar", "sueño"],
            "TRABAJO": ["trabajo", "laborar", "empleo"],
            "ESCUELA": ["escuela", "colegio", "institución educativa"],
            "MAMA": ["mamá", "madre", "progenitora"],
            "PAPA": ["papá", "padre", "progenitor"],
            "HERMANO": ["hermano", "hermano varón"],
            "HERMANA": ["hermana", "hermana mujer"],
            "ABUELO": ["abuelo", "abuelito"],
            "ABUELA": ["abuela", "abuelita"],
            "FELIZ": ["feliz", "alegre", "contento"],
            "TRISTE": ["triste", "melancólico", "deprimido"],
            "BUENO": ["bueno", "bien", "correcto"],
            "MALO": ["malo", "incorrecto", "erróneo"],
            "GRANDE": ["grande", "amplio", "extenso"],
            "PEQUEÑO": ["pequeño", "chico", "diminuto"],
            "ROJO": ["rojo", "color rojo"],
            "AZUL": ["azul", "color azul"],
            "VERDE": ["verde", "color verde"],
            "AMARILLO": ["amarillo", "color amarillo"],
            "NEGRO": ["negro", "color negro"],
            "BLANCO": ["blanco", "color blanco"],
            "CALIENTE": ["caliente", "calor", "temperatura alta"],
            "FRIO": ["frío", "helado", "temperatura baja"],
            "SOL": ["sol", "astro solar", "día soleado"],
            "DULCE": ["dulce", "azucarado", "sabor dulce"],
            "MEDICINA": ["medicina", "medicamento", "tratamiento"],
            "GATO": ["gato", "felino", "minino"],
            "PERRO": ["perro", "can", "mascota"],
            "CARRO": ["carro", "automóvil", "vehículo"],
            "CAMINAR": ["caminar", "andar", "pasear"],
            "LLORAR": ["llorar", "lágrimas", "tristeza"],
            "REIR": ["reír", "risa", "carcajada"],
            "COCINA": ["cocina", "lugar de cocinar"],
            "CAMA": ["cama", "lugar de dormir"],
            "LECHE": ["leche", "producto lácteo"],
            "YO": ["yo", "primera persona"],
            "USTED": ["usted", "segunda persona formal"]
        }
        
        for gesture in self.db.get_all_gestures():
            gesture_name = gesture["name"]
            if gesture_name in translations_dict:
                for translation in translations_dict[gesture_name]:
                    self.db.add_translation(gesture["id"], translation, "es")

def main():
    downloader = LSC50Downloader()
    
    # Verificar si necesitamos descargar
    logger.info("Iniciando descarga e importación del dataset LSC50...")
    
    # Descargar dataset
    if not downloader.download_dataset():
        logger.error("Falló la descarga del dataset")
        return
    
    # Extraer dataset
    if not downloader.extract_dataset():
        logger.error("Falló la extracción del dataset")
        return
    
    # Importar a base de datos
    if not downloader.import_to_database():
        logger.error("Falló la importación a la base de datos")
        return
    
    # Crear traducciones adicionales
    downloader.create_gesture_translations()
    
    logger.info("¡Dataset LSC50 importado exitosamente!")
    logger.info(f"Se importaron {len(downloader.lsc50_signs)} señas de LSC")
    
    # Mostrar estadísticas
    gestures = downloader.db.get_all_gestures()
    total_sequences = sum(len(downloader.db.get_gesture_sequences(g["id"])) for g in gestures)
    
    logger.info(f"Total de gestos en la base de datos: {len(gestures)}")
    logger.info(f"Total de secuencias: {total_sequences}")

if __name__ == "__main__":
    main() 