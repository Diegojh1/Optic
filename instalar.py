# Instalador automático para el Traductor LSC
# Guarda este archivo como "instalar.py" en tu carpeta C:\Users\diego\Proyectos\Optic

import subprocess
import sys
import os
from pathlib import Path

def ejecutar_comando(comando, descripcion):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n📦 {descripcion}...")
    print(f"Ejecutando: {comando}")
    
    try:
        result = subprocess.run(
            comando, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ {descripcion} - ÉXITO")
        if result.stdout:
            print(f"Salida: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {descripcion} - ERROR")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Salida: {e.stdout}")
        if e.stderr:
            print(f"Error detallado: {e.stderr}")
        return False

def instalar_dependencias():
    """Instala todas las dependencias necesarias"""
    print("🚀 INSTALANDO DEPENDENCIAS DEL TRADUCTOR LSC")
    print("=" * 60)
    
    # Actualizar pip primero
    ejecutar_comando(
        f'"{sys.executable}" -m pip install --upgrade pip',
        "Actualizando pip"
    )
    
    # Lista de dependencias con versiones específicas
    dependencias = [
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.7",
        "customtkinter==5.2.0",
        "Pillow==10.0.1",
        "numpy==1.24.3",
        "scipy==1.11.4",
        "pygame==2.5.2",
        "requests==2.31.0"
    ]
    
    print(f"\n📋 Instalando {len(dependencias)} dependencias...")
    
    for i, dependencia in enumerate(dependencias, 1):
        print(f"\n[{i}/{len(dependencias)}] Instalando {dependencia}")
        
        if not ejecutar_comando(
            f'"{sys.executable}" -m pip install {dependencia}',
            f"Instalación de {dependencia}"
        ):
            print(f"⚠️  Continuando con la siguiente dependencia...")
    
    print("\n✅ INSTALACIÓN DE DEPENDENCIAS COMPLETADA")

def crear_estructura_directorios():
    """Crea la estructura de directorios necesaria"""
    print("\n📁 CREANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 50)
    
    directorios = [
        "data",
        "data/cache",
        "data/gestures",
        "data/captures", 
        "data/exports",
        "data/sounds",
        "data/gifs"
    ]
    
    directorio_base = Path.cwd()
    print(f"Directorio base: {directorio_base}")
    
    for directorio in directorios:
        ruta = directorio_base / directorio
        try:
            ruta.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directorio}")
        except Exception as e:
            print(f"❌ Error creando {directorio}: {e}")
    
    print("\n✅ ESTRUCTURA DE DIRECTORIOS CREADA")

def crear_archivos_ejemplo():
    """Crea archivos de ejemplo y configuración"""
    print("\n📄 CREANDO ARCHIVOS DE CONFIGURACIÓN")
    print("=" * 50)
    
    # Crear config.json ejemplo
    config_ejemplo = '''{
  "zoom_speed": 3.0,
  "movement_threshold": 0.02,
  "api_cooldown": 2.0,
  "sound_effects_enabled": true,
  "ai_precision_mode": "balanced",
  "camera_index": 0,
  "camera_resolution": [640, 480],
  "frame_rate": 30,
  "synonyms": {
    "hola": ["saludar", "saludos", "buenos días", "buen día", "qué tal", "hey"],
    "adios": ["chao", "hasta luego", "despedida", "nos vemos", "hasta pronto"]
  }
}'''
    
    try:
        with open("data/config.json", "w", encoding="utf-8") as f:
            f.write(config_ejemplo)
        print("✅ config.json creado")
    except Exception as e:
        print(f"❌ Error creando config.json: {e}")
    
    # Crear README con instrucciones
    readme = '''# Traductor LSC Bidireccional

## Instalación Completada ✅

### Estructura de Archivos
- `main.py` - Archivo principal del traductor
- `data/` - Directorio de datos
- `data/gifs/` - Coloca aquí los archivos GIF de señas
- `data/sounds/` - Efectos de sonido (opcional)
- `data/captures/` - Capturas de pantalla
- `data/exports/` - Datos exportados

### Cómo Ejecutar
1. Asegúrate de estar en la carpeta del proyecto
2. Ejecuta: `python main.py`

### Agregar GIFs de Señas
1. Coloca los archivos .gif en la carpeta `data/gifs/`
2. Nombra los archivos según la palabra (ejemplo: `hola.gif`)
3. Edita el diccionario `text_to_sign_dict` en main.py

### Problemas Comunes
- Si no funciona la cámara: Verifica que no esté siendo usada por otra aplicación
- Si faltan dependencias: Ejecuta `pip install -r requirements.txt`
- Si no aparece la interfaz: Verifica que CustomTkinter esté instalado

### Soporte
Para problemas o mejoras, revisa el código y la documentación incluida.
'''
    
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme)
        print("✅ README.md creado")
    except Exception as e:
        print(f"❌ Error creando README.md: {e}")
    
    # Crear requirements.txt
    requirements = '''opencv-python==4.8.1.78
mediapipe==0.10.7
customtkinter==5.2.0
Pillow==10.0.1
numpy==1.24.3
scipy==1.11.4
pygame==2.5.2
requests==2.31.0
'''
    
    try:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements)
        print("✅ requirements.txt creado")
    except Exception as e:
        print(f"❌ Error creando requirements.txt: {e}")

def verificar_instalacion():
    """Verifica que todo esté instalado correctamente"""
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 40)
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar dependencias críticas
    dependencias_criticas = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('customtkinter', 'CustomTkinter'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy')
    ]
    
    todo_ok = True
    
    for modulo, nombre in dependencias_criticas:
        try:
            if modulo == 'PIL':
                from PIL import Image
                print(f"✅ {nombre}")
            else:
                __import__(modulo)
                print(f"✅ {nombre}")
        except ImportError:
            print(f"❌ {nombre} - NO INSTALADO")
            todo_ok = False
    
    # Verificar estructura de directorios
    directorios_requeridos = ["data", "data/gifs", "data/sounds"]
    for directorio in directorios_requeridos:
        if os.path.exists(directorio):
            print(f"✅ Directorio {directorio}")
        else:
            print(f"❌ Directorio {directorio} - NO EXISTE")
            todo_ok = False
    
    if todo_ok:
        print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("\nPara ejecutar el traductor:")
        print("1. Asegúrate de que main.py esté en esta carpeta")
        print("2. Ejecuta: python main.py")
    else:
        print("\n⚠️  Hay algunos problemas en la instalación")
        print("Revisa los errores arriba y ejecuta el instalador nuevamente")
    
    return todo_ok

def main():
    """Función principal del instalador"""
    print("🎯 INSTALADOR AUTOMÁTICO - TRADUCTOR LSC")
    print("=" * 60)
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Python: {sys.version}")
    
    input("\nPresiona Enter para comenzar la instalación...")
    
    try:
        # Paso 1: Instalar dependencias
        instalar_dependencias()
        
        # Paso 2: Crear estructura de directorios
        crear_estructura_directorios()
        
        # Paso 3: Crear archivos de configuración
        crear_archivos_ejemplo()
        
        # Paso 4: Verificar instalación
        if verificar_instalacion():
            print("\n🚀 ¡LISTO PARA USAR!")
        else:
            print("\n🔧 Revisa los errores y ejecuta nuevamente")
            
    except KeyboardInterrupt:
        print("\n❌ Instalación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la instalación: {e}")
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()