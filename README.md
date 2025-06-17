# Sistema de Traducción de Lengua de Señas Colombiana (LSC)

Este proyecto implementa un sistema de traducción bidireccional entre la Lengua de Señas Colombiana (LSC) y texto, utilizando visión por computadora y aprendizaje profundo.

## Características

- Recolección de datos de gestos LSC mediante cámara web
- Base de datos SQLite para almacenar gestos, secuencias y traducciones
- Modelo de reconocimiento de gestos basado en LSTM con mecanismo de atención
- Traducción en tiempo real de gestos a texto
- Interfaz visual con feedback en tiempo real
- Sistema de gestión de gestos y traducciones

## Requisitos

- Python 3.8+
- Cámara web
- GPU recomendada para entrenamiento (opcional)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/lsc-translator.git
cd lsc-translator
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
lsc-translator/
├── data/
│   └── gestures.db       # Base de datos de gestos
├── models/
│   ├── gesture_model.py  # Modelo de reconocimiento de gestos
│   └── gesture_model.pth # Modelo entrenado
├── database/
│   └── gesture_db.py     # Módulo de gestión de base de datos
├── collect_data.py       # Script para recolectar datos
├── train_model.py        # Script para entrenar el modelo
├── main.py              # Script principal de traducción
├── manage_gestures.py   # Script para gestionar gestos
└── requirements.txt     # Dependencias del proyecto
```

## Uso

### 1. Recolectar Datos

Para recolectar datos de gestos LSC:

```bash
python collect_data.py
```

Instrucciones:
- Presiona 'r' para iniciar/detener la grabación de un gesto
- Presiona 'd' para eliminar la última secuencia grabada
- Presiona 'q' para salir

### 2. Gestionar Gestos

Para gestionar la base de datos de gestos:

```bash
# Listar todos los gestos
python manage_gestures.py --action list

# Buscar gestos
python manage_gestures.py --action search --query "hola"

# Agregar traducción
python manage_gestures.py --action add_translation --gesture_id 1 --text "hola"

# Eliminar gesto
python manage_gestures.py --action delete --gesture_id 1
```

### 3. Entrenar Modelo

Para entrenar el modelo de reconocimiento:

```bash
python train_model.py --num_epochs 50 --batch_size 32
```

### 4. Traducir en Tiempo Real

Para iniciar el sistema de traducción:

```bash
python main.py
```

Instrucciones:
- Realiza gestos frente a la cámara
- El sistema mostrará la traducción en tiempo real
- Presiona 'q' para salir

## Base de Datos

El sistema utiliza una base de datos SQLite para almacenar:
- Gestos (nombre, descripción, categoría)
- Secuencias de gestos (landmarks, confianza)
- Traducciones (texto, idioma)

## Modelo de Reconocimiento

El modelo de reconocimiento de gestos incluye:
- Normalización de datos
- LSTM bidireccional
- Mecanismo de atención
- Capas fully connected
- Dropout para regularización

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Tu Nombre - [@tutwitter](https://twitter.com/tutwitter) - email@example.com

Link del Proyecto: [https://github.com/tu-usuario/lsc-translator](https://github.com/tu-usuario/lsc-translator) 