import argparse
import logging
from database.gesture_db import GestureDatabase
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_gestures(db: GestureDatabase):
    """Listar todos los gestos en la base de datos"""
    gestures = db.get_all_gestures()
    if not gestures:
        logger.info("No hay gestos en la base de datos")
        return
    
    logger.info("\nGestos en la base de datos:")
    logger.info("-" * 50)
    for gesture in gestures:
        sequences = db.get_gesture_sequences(gesture["id"])
        translations = db.get_gesture_translations(gesture["id"])
        
        logger.info(f"ID: {gesture['id']}")
        logger.info(f"Nombre: {gesture['name']}")
        logger.info(f"Descripción: {gesture['description']}")
        logger.info(f"Categoría: {gesture['category']}")
        logger.info(f"Secuencias: {len(sequences)}")
        logger.info(f"Traducciones: {', '.join(translations)}")
        logger.info("-" * 50)

def add_translation(db: GestureDatabase, gesture_id: int, text: str, language: str = "es"):
    """Agregar una traducción a un gesto"""
    try:
        translation_id = db.add_translation(gesture_id, text, language)
        logger.info(f"Traducción agregada (ID: {translation_id})")
    except Exception as e:
        logger.error(f"Error al agregar traducción: {e}")

def delete_gesture(db: GestureDatabase, gesture_id: int):
    """Eliminar un gesto y todas sus secuencias y traducciones"""
    try:
        if db.delete_gesture(gesture_id):
            logger.info(f"Gesto {gesture_id} eliminado")
        else:
            logger.error(f"No se pudo eliminar el gesto {gesture_id}")
    except Exception as e:
        logger.error(f"Error al eliminar gesto: {e}")

def search_gestures(db: GestureDatabase, query: str):
    """Buscar gestos por nombre"""
    gestures = db.search_gesture_by_name(query)
    if not gestures:
        logger.info(f"No se encontraron gestos que coincidan con '{query}'")
        return
    
    logger.info(f"\nResultados de búsqueda para '{query}':")
    logger.info("-" * 50)
    for gesture in gestures:
        sequences = db.get_gesture_sequences(gesture["id"])
        translations = db.get_gesture_translations(gesture["id"])
        
        logger.info(f"ID: {gesture['id']}")
        logger.info(f"Nombre: {gesture['name']}")
        logger.info(f"Descripción: {gesture['description']}")
        logger.info(f"Categoría: {gesture['category']}")
        logger.info(f"Secuencias: {len(sequences)}")
        logger.info(f"Traducciones: {', '.join(translations)}")
        logger.info("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Gestionar base de datos de gestos LSC")
    parser.add_argument("--db_path", default="data/gestures.db", help="Ruta a la base de datos")
    parser.add_argument("--action", choices=["list", "add_translation", "delete", "search"], required=True,
                      help="Acción a realizar")
    parser.add_argument("--gesture_id", type=int, help="ID del gesto")
    parser.add_argument("--text", help="Texto de traducción")
    parser.add_argument("--language", default="es", help="Idioma de la traducción")
    parser.add_argument("--query", help="Término de búsqueda")
    
    args = parser.parse_args()
    
    # Verificar que la base de datos existe
    if not os.path.exists(args.db_path):
        logger.error(f"No se encontró la base de datos en {args.db_path}")
        return
    
    # Inicializar base de datos
    db = GestureDatabase(args.db_path)
    
    # Ejecutar acción
    if args.action == "list":
        list_gestures(db)
    
    elif args.action == "add_translation":
        if not args.gesture_id or not args.text:
            logger.error("Se requiere gesture_id y text para agregar una traducción")
            return
        add_translation(db, args.gesture_id, args.text, args.language)
    
    elif args.action == "delete":
        if not args.gesture_id:
            logger.error("Se requiere gesture_id para eliminar un gesto")
            return
        delete_gesture(db, args.gesture_id)
    
    elif args.action == "search":
        if not args.query:
            logger.error("Se requiere query para buscar gestos")
            return
        search_gestures(db, args.query)

if __name__ == "__main__":
    main() 