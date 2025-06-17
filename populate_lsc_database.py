import logging
import numpy as np
from database.gesture_db import GestureDatabase
import json
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSCDatabasePopulator:
    def __init__(self, db_path: str = "data/gestures.db"):
        self.db = GestureDatabase(db_path)
        
        # Diccionario básico de LSC con traducciones
        self.lsc_gestures = {
            "HOLA": {
                "description": "Saludo básico en LSC",
                "category": "saludos",
                "translations": ["hola", "saludo", "saludar", "buenos días"]
            },
            "GRACIAS": {
                "description": "Expresión de gratitud",
                "category": "cortesía",
                "translations": ["gracias", "agradecer", "agradecimiento", "muchas gracias"]
            },
            "POR FAVOR": {
                "description": "Expresión de cortesía para solicitar",
                "category": "cortesía",
                "translations": ["por favor", "favor", "solicitar"]
            },
            "DISCULPE": {
                "description": "Expresión para disculparse",
                "category": "cortesía",
                "translations": ["disculpe", "perdón", "disculpa", "lo siento"]
            },
            "SI": {
                "description": "Afirmación positiva",
                "category": "respuestas",
                "translations": ["sí", "afirmativo", "correcto", "está bien"]
            },
            "NO": {
                "description": "Negación",
                "category": "respuestas",
                "translations": ["no", "negativo", "incorrecto", "no está bien"]
            },
            "MAMA": {
                "description": "Madre",
                "category": "familia",
                "translations": ["mamá", "madre", "progenitora", "mami"]
            },
            "PAPA": {
                "description": "Padre",
                "category": "familia",
                "translations": ["papá", "padre", "progenitor", "papi"]
            },
            "HERMANO": {
                "description": "Hermano varón",
                "category": "familia",
                "translations": ["hermano", "hermano varón", "franito"]
            },
            "HERMANA": {
                "description": "Hermana mujer",
                "category": "familia",
                "translations": ["hermana", "hermana mujer", "franita"]
            },
            "ABUELO": {
                "description": "Abuelo",
                "category": "familia",
                "translations": ["abuelo", "abuelito", "tata"]
            },
            "ABUELA": {
                "description": "Abuela",
                "category": "familia",
                "translations": ["abuela", "abuelita", "nana"]
            },
            "CASA": {
                "description": "Hogar o vivienda",
                "category": "lugares",
                "translations": ["casa", "hogar", "vivienda", "domicilio"]
            },
            "ESCUELA": {
                "description": "Institución educativa",
                "category": "lugares",
                "translations": ["escuela", "colegio", "universidad", "institución educativa"]
            },
            "TRABAJO": {
                "description": "Lugar de trabajo o actividad laboral",
                "category": "lugares",
                "translations": ["trabajo", "empleo", "laborar", "oficina"]
            },
            "HOSPITAL": {
                "description": "Centro médico",
                "category": "lugares",
                "translations": ["hospital", "clínica", "centro médico", "médico"]
            },
            "AGUA": {
                "description": "Líquido vital",
                "category": "alimentos",
                "translations": ["agua", "líquido", "beber agua", "hidratarse"]
            },
            "COMER": {
                "description": "Acción de alimentarse",
                "category": "acciones",
                "translations": ["comer", "alimentarse", "comida", "alimento"]
            },
            "BEBER": {
                "description": "Acción de tomar líquidos",
                "category": "acciones",
                "translations": ["beber", "tomar", "líquido", "hidratarse"]
            },
            "DORMIR": {
                "description": "Acción de descansar",
                "category": "acciones",
                "translations": ["dormir", "descansar", "sueño", "reposar"]
            },
            "ESTUDIAR": {
                "description": "Acción de aprender",
                "category": "acciones",
                "translations": ["estudiar", "aprender", "leer", "educarse"]
            },
            "CAMINAR": {
                "description": "Acción de moverse a pie",
                "category": "acciones",
                "translations": ["caminar", "andar", "pasear", "ir a pie"]
            },
            "FELIZ": {
                "description": "Estado emocional positivo",
                "category": "emociones",
                "translations": ["feliz", "alegre", "contento", "gozoso"]
            },
            "TRISTE": {
                "description": "Estado emocional negativo",
                "category": "emociones",
                "translations": ["triste", "melancólico", "deprimido", "afligido"]
            },
            "ENOJADO": {
                "description": "Estado de molestia",
                "category": "emociones",
                "translations": ["enojado", "molesto", "furioso", "irritado"]
            },
            "MIEDO": {
                "description": "Sentimiento de temor",
                "category": "emociones",
                "translations": ["miedo", "temor", "susto", "pánico"]
            },
            "ROJO": {
                "description": "Color rojo",
                "category": "colores",
                "translations": ["rojo", "color rojo", "carmesí"]
            },
            "AZUL": {
                "description": "Color azul",
                "category": "colores",
                "translations": ["azul", "color azul", "celeste"]
            },
            "VERDE": {
                "description": "Color verde",
                "category": "colores",
                "translations": ["verde", "color verde", "esmeralda"]
            },
            "AMARILLO": {
                "description": "Color amarillo",
                "category": "colores",
                "translations": ["amarillo", "color amarillo", "dorado"]
            },
            "NEGRO": {
                "description": "Color negro",
                "category": "colores",
                "translations": ["negro", "color negro", "oscuro"]
            },
            "BLANCO": {
                "description": "Color blanco",
                "category": "colores",
                "translations": ["blanco", "color blanco", "claro"]
            },
            "GRANDE": {
                "description": "Tamaño amplio",
                "category": "descriptivos",
                "translations": ["grande", "amplio", "extenso", "enorme"]
            },
            "PEQUEÑO": {
                "description": "Tamaño reducido",
                "category": "descriptivos",
                "translations": ["pequeño", "chico", "diminuto", "tiny"]
            },
            "BUENO": {
                "description": "Calidad positiva",
                "category": "descriptivos",
                "translations": ["bueno", "bien", "correcto", "excelente"]
            },
            "MALO": {
                "description": "Calidad negativa",
                "category": "descriptivos",
                "translations": ["malo", "mal", "incorrecto", "pésimo"]
            },
            "YO": {
                "description": "Primera persona singular",
                "category": "pronombres",
                "translations": ["yo", "primera persona", "mí", "me"]
            },
            "TU": {
                "description": "Segunda persona singular",
                "category": "pronombres",
                "translations": ["tú", "usted", "segunda persona", "ti"]
            },
            "EL": {
                "description": "Tercera persona singular masculino",
                "category": "pronombres",
                "translations": ["él", "tercera persona", "le"]
            },
            "ELLA": {
                "description": "Tercera persona singular femenino",
                "category": "pronombres",
                "translations": ["ella", "tercera persona", "le"]
            },
            "NOSOTROS": {
                "description": "Primera persona plural",
                "category": "pronombres",
                "translations": ["nosotros", "nosotras", "nos"]
            },
            "QUE": {
                "description": "Interrogativo qué",
                "category": "interrogativos",
                "translations": ["qué", "que", "cuál"]
            },
            "QUIEN": {
                "description": "Interrogativo quién",
                "category": "interrogativos",
                "translations": ["quién", "quien", "quiénes"]
            },
            "DONDE": {
                "description": "Interrogativo dónde",
                "category": "interrogativos",
                "translations": ["dónde", "donde", "lugar"]
            },
            "CUANDO": {
                "description": "Interrogativo cuándo",
                "category": "interrogativos",
                "translations": ["cuándo", "cuando", "tiempo"]
            },
            "COMO": {
                "description": "Interrogativo cómo",
                "category": "interrogativos",
                "translations": ["cómo", "como", "manera"]
            },
            "POR QUE": {
                "description": "Interrogativo por qué",
                "category": "interrogativos",
                "translations": ["por qué", "porque", "razón"]
            },
            "UNO": {
                "description": "Número uno",
                "category": "números",
                "translations": ["uno", "1", "primero"]
            },
            "DOS": {
                "description": "Número dos",
                "category": "números",
                "translations": ["dos", "2", "segundo"]
            },
            "TRES": {
                "description": "Número tres",
                "category": "números",
                "translations": ["tres", "3", "tercero"]
            },
            "CUATRO": {
                "description": "Número cuatro",
                "category": "números",
                "translations": ["cuatro", "4", "cuarto"]
            },
            "CINCO": {
                "description": "Número cinco",
                "category": "números",
                "translations": ["cinco", "5", "quinto"]
            },
            "CALIENTE": {
                "description": "Temperatura alta",
                "category": "sensaciones",
                "translations": ["caliente", "calor", "temperatura alta"]
            },
            "FRIO": {
                "description": "Temperatura baja",
                "category": "sensaciones",
                "translations": ["frío", "fresco", "temperatura baja"]
            },
            "DOLOR": {
                "description": "Sensación de dolor",
                "category": "sensaciones",
                "translations": ["dolor", "duele", "lastimar"]
            },
            "AYUDA": {
                "description": "Solicitud de asistencia",
                "category": "necesidades",
                "translations": ["ayuda", "auxilio", "asistencia", "socorro"]
            },
            "MEDICINA": {
                "description": "Medicamento o tratamiento",
                "category": "salud",
                "translations": ["medicina", "medicamento", "tratamiento", "pastilla"]
            },
            "MEDICO": {
                "description": "Profesional de la salud",
                "category": "salud",
                "translations": ["médico", "doctor", "doctora", "profesional salud"]
            }
        }
    
    def create_sample_sequence(self, gesture_name: str) -> np.ndarray:
        """Crear una secuencia de ejemplo para un gesto"""
        # Crear secuencia sintética de 30 frames con variación
        sequence_length = 30
        num_landmarks = 63  # 21 landmarks * 3 coordenadas por mano (asumiendo una mano)
        
        # Generar secuencia base con patrón específico para cada gesto
        base_pattern = np.random.rand(sequence_length, num_landmarks) * 0.1
        
        # Agregar patrón específico basado en el hash del nombre del gesto
        gesture_hash = hash(gesture_name) % 1000
        pattern_modifier = np.sin(np.linspace(0, 2*np.pi*gesture_hash/100, sequence_length))
        
        for i in range(sequence_length):
            base_pattern[i] += pattern_modifier[i] * 0.05
        
        # Normalizar valores entre 0 y 1
        base_pattern = np.clip(base_pattern, 0, 1)
        
        return base_pattern.astype(np.float32)
    
    def populate_database(self):
        """Poblar la base de datos con gestos LSC"""
        logger.info("Iniciando población de la base de datos con gestos LSC...")
        
        for gesture_name, gesture_info in self.lsc_gestures.items():
            # Crear gesto en la base de datos
            gesture_id = self.db.add_gesture(
                name=gesture_name,
                description=gesture_info["description"],
                category=gesture_info["category"]
            )
            
            # Agregar traducciones
            for translation in gesture_info["translations"]:
                self.db.add_translation(gesture_id, translation, "es")
            
            # Crear múltiples secuencias de ejemplo para cada gesto
            for i in range(5):  # 5 secuencias por gesto
                sequence = self.create_sample_sequence(f"{gesture_name}_{i}")
                confidence = 0.85 + (np.random.rand() * 0.1)  # Confianza entre 0.85 y 0.95
                self.db.add_sequence(gesture_id, sequence, confidence)
            
            logger.info(f"Agregado gesto: {gesture_name} con {len(gesture_info['translations'])} traducciones")
        
        logger.info("¡Base de datos poblada exitosamente!")
        
        # Mostrar estadísticas
        gestures = self.db.get_all_gestures()
        total_sequences = sum(len(self.db.get_gesture_sequences(g["id"])) for g in gestures)
        total_translations = sum(len(self.db.get_gesture_translations(g["id"])) for g in gestures)
        
        logger.info(f"Total de gestos: {len(gestures)}")
        logger.info(f"Total de secuencias: {total_sequences}")
        logger.info(f"Total de traducciones: {total_translations}")
        
        # Agrupar por categorías
        categories = {}
        for gesture in gestures:
            category = gesture["category"]
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        logger.info("Gestos por categoría:")
        for category, count in categories.items():
            logger.info(f"  {category}: {count} gestos")
    
    def save_gesture_mapping(self):
        """Guardar mapeo de gestos para referencia"""
        mapping_file = "data/lsc_gesture_mapping.json"
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        
        gestures = self.db.get_all_gestures()
        mapping = {}
        
        for gesture in gestures:
            translations = self.db.get_gesture_translations(gesture["id"])
            mapping[gesture["name"]] = {
                "id": gesture["id"],
                "description": gesture["description"],
                "category": gesture["category"],
                "translations": translations
            }
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Mapeo de gestos guardado en: {mapping_file}")

def main():
    # Crear directorio de datos si no existe
    os.makedirs("data", exist_ok=True)
    
    # Crear poblador de base de datos
    populator = LSCDatabasePopulator()
    
    # Poblar base de datos
    populator.populate_database()
    
    # Guardar mapeo de gestos
    populator.save_gesture_mapping()
    
    logger.info("¡Sistema LSC listo para usar!")
    logger.info("Ahora puedes:")
    logger.info("1. Recolectar más datos: python collect_data.py")
    logger.info("2. Entrenar el modelo: python train_model.py")
    logger.info("3. Usar el traductor: python main.py")

if __name__ == "__main__":
    main() 