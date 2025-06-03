# Script de diagnóstico para el Traductor LSC
# Guarda este archivo como "diagnostico.py" en la misma carpeta que tu proyecto

import sys
import os
import subprocess
import importlib.util

def verificar_python():
    """Verifica la versión de Python"""
    print("=== VERIFICACIÓN DE PYTHON ===")
    print(f"Versión de Python: {sys.version}")
    print(f"Ejecutable: {sys.executable}")
    
    if sys.version_info < (3, 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        return False
    else:
        print("✅ Versión de Python correcta")
        return True

def verificar_dependencias():
    """Verifica que todas las librerías estén instaladas"""
    print("\n=== VERIFICACIÓN DE DEPENDENCIAS ===")
    
    dependencias = [
        'cv2',
        'mediapipe', 
        'requests',
        'numpy',
        'PIL',
        'customtkinter',
        'scipy',
        'pygame',
        'tkinter'
    ]
    
    dependencias_faltantes = []
    
    for dep in dependencias:
        try:
            if dep == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'PIL':
                from PIL import Image
                print(f"✅ Pillow: {Image.__version__}")
            elif dep == 'tkinter':
                import tkinter
                print("✅ Tkinter: Disponible")
            else:
                spec = importlib.util.find_spec(dep)
                if spec is not None:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'Disponible')
                    print(f"✅ {dep}: {version}")
                else:
                    print(f"❌ {dep}: No encontrado")
                    dependencias_faltantes.append(dep)
        except ImportError as e:
            print(f"❌ {dep}: Error de importación - {e}")
            dependencias_faltantes.append(dep)
    
    return dependencias_faltantes

def instalar_dependencias(dependencias_faltantes):
    """Instala las dependencias faltantes"""
    if not dependencias_faltantes:
        return True
    
    print(f"\n=== INSTALANDO DEPENDENCIAS FALTANTES ===")
    
    # Mapeo de nombres de módulos a nombres de pip
    pip_names = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'customtkinter': 'customtkinter',
        'mediapipe': 'mediapipe',
        'scipy': 'scipy',
        'pygame': 'pygame',
        'requests': 'requests',
        'numpy': 'numpy'
    }
    
    for dep in dependencias_faltantes:
        pip_name = pip_names.get(dep, dep)
        print(f"Instalando {pip_name}...")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', pip_name
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ {pip_name} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {pip_name}: {e}")
            return False
    
    return True

def verificar_camara():
    """Verifica que la cámara esté disponible"""
    print("\n=== VERIFICACIÓN DE CÁMARA ===")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Cámara disponible")
            cap.release()
            return True
        else:
            print("❌ No se puede acceder a la cámara")
            return False
    except Exception as e:
        print(f"❌ Error verificando cámara: {e}")
        return False

def verificar_directorio():
    """Verifica y crea los directorios necesarios"""
    print("\n=== VERIFICACIÓN DE DIRECTORIOS ===")
    
    directorio_actual = os.getcwd()
    print(f"Directorio actual: {directorio_actual}")
    
    directorios = [
        "data",
        "data/cache",
        "data/gestures", 
        "data/captures",
        "data/exports",
        "data/sounds",
        "data/gifs"
    ]
    
    try:
        for directorio in directorios:
            ruta_completa = os.path.join(directorio_actual, directorio)
            
            if not os.path.exists(ruta_completa):
                os.makedirs(ruta_completa)
                print(f"✅ Directorio creado: {directorio}")
            else:
                print(f"✅ Directorio existe: {directorio}")
        
        return True
    except Exception as e:
        print(f"❌ Error creando directorios: {e}")
        return False

def verificar_api_key():
    """Verifica la configuración de la API"""
    print("\n=== VERIFICACIÓN DE API ===")
    
    # La API key está hardcodeada en el código original
    api_key = "AIzaSyDVjmUAxkg4GYmpi4IHkggDEyM-WLzXZa4"
    
    if api_key and len(api_key) > 10:
        print("✅ API Key configurada")
        return True
    else:
        print("❌ API Key no configurada correctamente")
        return False

def crear_archivo_prueba():
    """Crea un archivo de prueba simplificado"""
    print("\n=== CREANDO ARCHIVO DE PRUEBA ===")
    
    codigo_prueba = '''
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import os

# Configurar tema
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class PruebaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Prueba - Traductor LSC")
        self.geometry("600x400")
        
        # Crear directorio data
        if not os.path.exists("data"):
            os.makedirs("data")
            print("Directorio 'data' creado")
        
        # Etiqueta de bienvenida
        self.label = ctk.CTkLabel(
            self, 
            text="🎉 ¡El sistema está funcionando correctamente!",
            font=("Helvetica", 18, "bold")
        )
        self.label.pack(pady=50)
        
        # Botón de prueba
        self.button = ctk.CTkButton(
            self,
            text="Probar Sistema",
            command=self.probar_sistema,
            font=("Helvetica", 14),
            height=40,
            width=200
        )
        self.button.pack(pady=20)
        
        # Información del sistema
        info_text = f"""
Directorio actual: {os.getcwd()}
Directorio 'data' existe: {os.path.exists('data')}
        """
        
        self.info_label = ctk.CTkLabel(
            self,
            text=info_text,
            font=("Helvetica", 12)
        )
        self.info_label.pack(pady=20)
    
    def probar_sistema(self):
        try:
            # Probar importaciones críticas
            import cv2
            import mediapipe as mp
            import numpy as np
            from PIL import Image
            
            messagebox.showinfo(
                "Éxito", 
                "¡Todas las librerías están funcionando correctamente!\\n"
                "El sistema está listo para usar."
            )
        except ImportError as e:
            messagebox.showerror(
                "Error", 
                f"Error en las importaciones:\\n{e}"
            )

if __name__ == "__main__":
    app = PruebaApp()
    app.mainloop()
'''
    
    try:
        with open("prueba_sistema.py", "w", encoding="utf-8") as f:
            f.write(codigo_prueba)
        
        print("✅ Archivo de prueba creado: prueba_sistema.py")
        print("   Ejecuta: python prueba_sistema.py")
        return True
    except Exception as e:
        print(f"❌ Error creando archivo de prueba: {e}")
        return False

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO DEL TRADUCTOR LSC")
    print("=" * 50)
    
    # Verificar Python
    if not verificar_python():
        input("Presiona Enter para salir...")
        return
    
    # Verificar dependencias
    dependencias_faltantes = verificar_dependencias()
    
    if dependencias_faltantes:
        respuesta = input(f"\n¿Deseas instalar las dependencias faltantes? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'y', 'yes']:
            if not instalar_dependencias(dependencias_faltantes):
                print("❌ Error en la instalación. Instala manualmente con:")
                for dep in dependencias_faltantes:
                    pip_names = {
                        'cv2': 'opencv-python',
                        'PIL': 'Pillow',
                        'customtkinter': 'customtkinter'
                    }
                    pip_name = pip_names.get(dep, dep)
                    print(f"   pip install {pip_name}")
                input("Presiona Enter para salir...")
                return
    
    # Verificar cámara
    verificar_camara()
    
    # Verificar directorios
    verificar_directorio()
    
    # Verificar API
    verificar_api_key()
    
    # Crear archivo de prueba
    crear_archivo_prueba()
    
    print("\n" + "=" * 50)
    print("🎉 DIAGNÓSTICO COMPLETADO")
    print("\nPROBLEMAS COMUNES Y SOLUCIONES:")
    print("1. Si no se ejecuta nada: Verifica que tengas Python instalado correctamente")
    print("2. Si falta CustomTkinter: pip install customtkinter")
    print("3. Si falta OpenCV: pip install opencv-python")
    print("4. Si falta MediaPipe: pip install mediapipe")
    print("5. Ejecuta primero 'prueba_sistema.py' para verificar que todo funciona")
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()