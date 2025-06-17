# 🖐️ OPTIC: Sistema de Traducción Bidireccional de Lenguaje de Señas

**Ocular Powered Translation & Interpretation Core**

Un sistema avanzado de reconocimiento y traducción de lengua de señas que combina inteligencia artificial con aprendizaje personalizado para ofrecer traducciones precisas en tiempo real, potenciado por controles oculares innovadores.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Características Principales

### 🔄 **Traducción Bidireccional**
- **Señas → Texto**: Reconocimiento en tiempo real usando IA avanzada
- **Texto → Señas**: Conversión a animaciones GIF de señas
- **Sistema Híbrido**: Combina gestos entrenados personalmente con IA

### 🧠 **Inteligencia Artificial**
- **Gemini API 2.0**: Modelo multimodal de Google para reconocimiento
- **Visión Artificial**: MediaPipe para detección precisa de manos y rostro
- **Aprendizaje Personalizado**: Entrena tus propios gestos para mayor precisión

### 👁️ **Controles Avanzados - OPTIC Core**
- **Zoom por Gestos**: Separa pulgar e índice para controlar zoom
- **Controles por Parpadeo**: 
  - Modo normal: Zoom on/off
  - Modo entrenamiento: Captura de gestos
- **Interfaz Intuitiva**: Controles tanto manuales como por gestos oculares

### ⚡ **Optimización y Estabilidad**
- **Sistema de Cooldown**: Evita traducciones erróneas
- **Estabilización de Gestos**: Requiere consistencia antes de traducir
- **Prioridad Inteligente**: Gestos entrenados > IA > Respaldo

## 🚀 Instalación

### Requisitos del Sistema
- **Python**: 3.11 o superior
- **Webcam**: Resolución mínima 720p
- **RAM**: 8GB mínimo recomendado
- **Conexión**: Internet para API de IA
- **OS**: Windows 10/11, macOS 12+, Ubuntu 20.04+

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Diegojh1/Optic.git
cd Optic
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Instalar Dependencias

**Opción A: Instalación Automática (Recomendada)**
```bash
python install.py
```

**Opción B: Instalación Manual**
```bash
pip install -r requirements.txt
```

### 4. Configurar API Key
1. Obtén tu API key de [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Edita `main.py` y reemplaza tu API key:
```python
GEMINI_API_KEY = "TU_API_KEY_AQUI"
```

### 5. Crear Estructura de Directorios
```bash
mkdir -p data/{cache,gestures,captures,exports,sounds,gifs}
```

### 6. Agregar Archivos de Audio (Opcional)
Coloca archivos MP3 en `data/sounds/`:
- `scan.mp3` - Sonido de inicio
- `beep.mp3` - Sonido de comandos
- `success.mp3` - Sonido de éxito
- `capture.mp3` - Sonido de captura
- `alert.mp3` - Sonido de alerta
- `error.mp3` - Sonido de error
- `click.mp3` - Sonido de clic

### 7. Agregar GIFs de Señas (Opcional)
Coloca archivos GIF en `data/gifs/`:
- `hola.gif`
- `gracias.gif`
- `buenos_dias.gif`
- etc.

## 🎮 Uso

### Iniciar la Aplicación
```bash
python main.py
```

### 🖐️ **Modo: Señas a Texto**
1. **Iniciar Cámara**: Presiona "▶️ INICIAR CÁMARA"
2. **Posicionar Manos**: Espera a que se detecten tus manos
3. **Formar Gesto**: El sistema esperará a que tu mano esté estable
4. **Ver Traducción**: Aparecerá en la parte inferior de la pantalla

**Controles Especiales OPTIC:**
- 👁️ **Parpadeo (1s)**: Activar/desactivar zoom
- 🤏 **Pulgar + Índice**: Controlar nivel de zoom

### 📝 **Modo: Texto a Señas**
1. **Escribir Texto**: En el campo de entrada
2. **Presionar "Traducir"**: El sistema buscará el GIF correspondiente
3. **Ver Animación**: Se mostrará el GIF de la seña

### 🎯 **Modo: Entrenamiento de Gestos**
1. **Nombrar Gesto**: Escribe el nombre en el campo
2. **Iniciar Entrenamiento**: Presiona "Iniciar Entrenamiento"
3. **Formar Gesto**: Realiza el gesto que quieres entrenar
4. **Capturar Muestras**: 
   - 📸 **Botón**: "Capturar Gesto"
   - 👁️ **Parpadeo**: Cierra ojos por 1 segundo (Tecnología OPTIC)
5. **Repetir**: Hasta tener al menos 10 muestras
6. **Finalizar**: Presiona "✅ FINALIZAR ENTRENAMIENTO"

### 📊 **Historial y Configuración**
- **Historial**: Ver todas las traducciones realizadas
- **Configuración**: Ajustar umbrales, tiempos, sensibilidad
- **Exportar**: Guardar historial de traducciones

## ⚙️ Configuración

### Parámetros Principales
| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| **Translation Cooldown** | Tiempo entre traducciones | 1.0 segundos |
| **Movement Threshold** | Sensibilidad de movimiento | 0.02 |
| **Gesture Recognition** | Umbral para gestos entrenados | 0.7 |
| **Frame Skip** | Procesar cada N frames | 3 |

### Personalización
- **Sonidos**: Reemplaza archivos en `data/sounds/`
- **GIFs**: Agrega nuevos en `data/gifs/`
- **Sinónimos**: Edita `synonym_dict` en `main.py`
- **API**: Cambia configuración de precisión (fast/balanced/precise)

## 📁 Estructura del Proyecto

```
optic/
├── main.py                 # Aplicación principal
├── README.md              # Este archivo
├── requirements.txt       # Dependencias Python
├── data/                  # Datos de la aplicación
│   ├── cache/            # Cache de traducciones
│   ├── gestures/         # Gestos entrenados (.json)
│   ├── captures/         # Capturas de pantalla
│   ├── exports/          # Datos exportados
│   ├── sounds/           # Efectos de sonido (.mp3)
│   ├── gifs/            # Animaciones de señas (.gif)
│   └── config.json      # Configuración guardada
└── venv/                 # Entorno virtual (creado automáticamente)
```

## 🔧 Tecnologías

### Core
- **Python 3.11+**: Lenguaje principal
- **OpenCV**: Procesamiento de imágenes
- **MediaPipe**: Detección de manos y rostro
- **NumPy/SciPy**: Análisis numérico

### Interfaz
- **CustomTkinter**: Interfaz gráfica moderna
- **PIL/Pillow**: Manipulación de imágenes
- **Pygame**: Efectos de sonido

### IA y API
- **Google Gemini 2.0**: Reconocimiento de gestos
- **Requests**: Comunicación con API
- **JSON**: Almacenamiento de datos

## 🎨 Funcionalidades Avanzadas

### Sistema de Estabilización
- **Preparación de Gestos**: 0.3s de preparación inicial
- **Verificación de Estabilidad**: 0.5s de mano estable
- **Buffer de Consistencia**: 3 detecciones consecutivas
- **Cooldown Inteligente**: Evita traducciones falsas

### Reconocimiento Híbrido
```
Flujo de Decisión:
1. ¿Gesto entrenado con confianza ≥ 30%? → Usar directamente
2. ¿No hay gesto entrenado? → Consultar IA
3. ¿Ambos fallan? → Sin traducción
```

### Controles Multimodales OPTIC
- **Visual**: Interfaz gráfica completa
- **Gestual**: Zoom y navegación por gestos
- **Ocular**: Comandos por parpadeo (Tecnología OPTIC Core)
- **Manual**: Botones tradicionales

## 🐛 Resolución de Problemas

### Problemas Comunes

**❌ "Error: No se puede acceder a la cámara"**
```bash
# Verificar cámaras disponibles
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**❌ "Error de API: 403 Forbidden"**
- Verifica tu API key de Gemini
- Confirma que tienes créditos disponibles
- Revisa límites de velocidad

**❌ "No se detectan gestos entrenados"**
- Verifica archivos `.json` en `data/gestures/`
- Ajusta umbral de reconocimiento en configuración
- Reentrena gestos con más muestras

**❌ "Interfaz se ve mal"**
- Actualiza `customtkinter`: `pip install --upgrade customtkinter`
- Verifica resolución de pantalla mínima: 1024x700

### Performance
- **FPS bajo**: Aumenta `processing_frame_skip` en configuración
- **Lag de video**: Reduce calidad de imagen o resolución de cámara
- **Traducciones lentas**: Ajusta `translation_cooldown`

## 🤝 Contribución

### Cómo Contribuir
1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crea** un Pull Request

### Áreas de Mejora
- 🎯 **Más Gestos**: Ampliar diccionario de señas
- 🌐 **Idiomas**: Soporte para otras lenguas de señas
- 📱 **Mobile**: Versión para dispositivos móviles
- 🔊 **TTS**: Síntesis de voz para traducciones
- 🎥 **Video**: Soporte para traducción de videos
- 👁️ **OPTIC Enhanced**: Más controles oculares avanzados

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Diego Hernández** - Desarrollador Principal
- GitHub: [@Diegojh1](https://github.com/Diegojh1)
- Instagram: [@diegojh_1](https://instagram.com/diegojh_1)

## 🙏 Agradecimientos

- **Google AI**: Por la API Gemini
- **MediaPipe Team**: Por las herramientas de visión artificial
- **Comunidad LSC**: Por feedback y validación de señas
- **OpenCV Community**: Por las herramientas de procesamiento de imagen

## 📊 Estadísticas del Proyecto

- **Líneas de código**: ~3,700
- **Funciones**: 50+
- **Modos de operación**: 4
- **Formatos soportados**: JSON, GIF, MP3, PNG
- **APIs integradas**: 1 (Gemini)
- **Tecnología OPTIC**: Controles oculares integrados

---

**⭐ Si te gusta este proyecto, ¡dale una estrella en GitHub!**

**🐛 Reporta bugs o 💡 sugiere mejoras en [Issues](https://github.com/Diegojh1/optic/issues)** 