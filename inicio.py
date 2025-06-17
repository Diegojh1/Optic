#!/usr/bin/env python3
"""
Script de inicio simplificado para el Traductor LSC
Guarda este archivo como "inicio.py" en tu carpeta C:\Users\diego\Proyectos\Optic
"""

import os
import sys
import subprocess
from pathlib import Path

def verificar_entorno():
    """Verifica que el entorno esté listo"""
    print("🔍 VERIFICANDO ENTORNO")
    print("=" * 40)
    
    # Verificar directorio actual
    directorio_actual = Path.cwd()
    print(f"📁 Directorio actual: {directorio_actual}")
    
    # Verificar Python
    print(f"🐍 Python: {sys.version}")
    
    # Verificar que existe main.py
    main_py = directorio_actual / "main.py"
    if main_py.exists():
        print("✅ main.py encontrado")
    else:
        print("❌ main.py NO encontrado")
        print("   Asegúrate de que main.py esté en esta carpeta")
        return False
    
    # Crear directorio data si no existe
    data_dir = directorio_actual / "data"
    if not data_dir.exists():
        try:
            data_dir.mkdir()
            print("✅ Directorio 'data' creado")
        except Exception as e:
            print(f"❌ Error creando directorio 'data': {e}")
            return False
    else:
        print("✅ Directorio 'data' existe")
    
    return True

def verificar_dependencias():
    """Verifica dependencias críticas"""
    print("\n🔧 VERIFICANDO DEPENDENCIAS CRÍTICAS")
    print("=" * 40)
    
    dependencias_criticas = [
        'customtkinter',
        'cv2',
        'mediapipe',
        'PIL'
    ]
    
    for dep in dependencias_criticas:
        try:
            if dep == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'PIL':
                from PIL import Image
                print(f"✅ Pillow: Disponible")
            else:
                __import__(dep)
                print(f"✅ {dep}: Disponible")
        except ImportError:
            print(f"❌ {dep}: NO INSTALADO")
            return False
    
    return True

def instalar_dependencias_faltantes():
    """Instala dependencias faltantes"""
    print("\n📦 INSTALANDO DEPENDENCIAS FALTANTES")
    print("=" * 40)
    
    dependencias = [
        'customtkinter',
        'opencv-python',
        'mediapipe',
        'Pillow',
        'numpy',
        'scipy',
        'pygame',
        'requests'
    ]
    
    for dep in dependencias:
        try:
            print(f"📦 Instalando {dep}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep
            ], capture_output=True, text=True, check=True)
            print(f"✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Error instalando {dep}: {e}")
    
    print("✅ Instalación completada")

def ejecutar_aplicacion():
    """Ejecuta la aplicación principal"""
    print("\n🚀 INICIANDO APLICACIÓN")
    print("=" * 40)
    
    try:
        # Importar y ejecutar la aplicación
        import main
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando aplicación: {e}")
        return False

def crear_archivos_basicos():
    """Crea archivos básicos necesarios"""
    print("\n📄 CREANDO ARCHIVOS BÁSICOS")
    print("=" * 40)
    
    # Crear subdirectorios en data
    subdirs = ['gifs', 'sounds', 'captures', 'exports', 'cache', 'gestures']
    
    for subdir in subdirs:
        path = Path('data') / subdir
        try:
            path.mkdir(exist_ok=True)
            print(f"✅ {subdir}/")
        except Exception as e:
            print(f"❌ Error creando {subdir}: {e}")
    
    # Crear archivo README en gifs
    gifs_readme = Path('data/gifs/README.txt')
    if not gifs_readme.exists():
        try:
            with open(gifs_readme, 'w', encoding='utf-8') as f:
                f.write("""
CARPETA DE GIFS - LENGUA DE SEÑAS
================================

📁 Coloca aquí los archivos .gif de las señas LSC

📝 NOMBRES RECOMENDADOS:
- hola.gif
- adios.gif  
- gracias.gif
- por_favor.gif
- si.gif
- no.gif

🎯 FORMATO:
- Archivos .gif animados
- Resolución recomendada: 400x400 píxeles
- Duración: 2-4 segundos

💡 CONSEJOS:
- Usa nombres descriptivos en minúsculas
- Evita espacios (usa guiones bajos _)
- Los GIFs con fondo transparente se ven mejor

Después de agregar GIFs, actualiza el diccionario
text_to_sign_dict en main.py
""")
            print("✅ README creado en gifs/")
        except Exception as e:
            print(f"❌ Error creando README: {e}")

def menu_principal():
    """Menú principal interactivo"""
    while True:
        print("\n" + "=" * 50)
        print("🎯 TRADUCTOR LSC BIDIRECCIONAL - MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. 🚀 Ejecutar aplicación")
        print("2. 🔧 Verificar entorno")
        print("3. 📦 Instalar dependencias")
        print("4. 📁 Crear estructura de archivos")
        print("5. ❌ Salir")
        print("=" * 50)
        
        opcion = input("Selecciona una opción (1-5): ").strip()
        
        if opcion == "1":
            if verificar_entorno() and verificar_dependencias():
                if ejecutar_aplicacion():
                    print("✅ Aplicación ejecutada correctamente")
                else:
                    print("❌ Error ejecutando la aplicación")
            else:
                print("❌ El entorno no está listo")
                respuesta = input("¿Deseas instalar dependencias? (s/n): ").strip().lower()
                if respuesta in ['s', 'si', 'y', 'yes']:
                    instalar_dependencias_faltantes()
        
        elif opcion == "2":
            verificar_entorno()
            verificar_dependencias()
        
        elif opcion == "3":
            instalar_dependencias_faltantes()
        
        elif opcion == "4":
            crear_archivos_basicos()
        
        elif opcion == "5":
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")
        
        input("\nPresiona Enter para continuar...")

def main():
    """Función principal"""
    try:
        print("🎯 TRADUCTOR LSC BIDIRECCIONAL")
        print("Directorio actual:", os.getcwd())
        print()
        
        # Verificación automática inicial
        if verificar_entorno() and verificar_dependencias():
            print("✅ Todo listo para ejecutar")
            respuesta = input("¿Ejecutar aplicación ahora? (s/n): ").strip().lower()
            
            if respuesta in ['s', 'si', 'y', 'yes', '']:
                ejecutar_aplicacion()
            else:
                menu_principal()
        else:
            print("⚠️  Hay problemas en el entorno")
            menu_principal()
    
    except KeyboardInterrupt:
        print("\n👋 Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()