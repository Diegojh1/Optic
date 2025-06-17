import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging
import argparse
from database.gesture_db import GestureDatabase
from models.gesture_model import GestureModel
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int], sequence_length: int = 30):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Asegurar que la secuencia tenga la longitud correcta
        if len(sequence) > self.sequence_length:
            # Tomar frames equiespaciados
            indices = np.linspace(0, len(sequence)-1, self.sequence_length, dtype=int)
            sequence = sequence[indices]
        elif len(sequence) < self.sequence_length:
            # Repetir el último frame
            last_frame = sequence[-1]
            padding = np.repeat(last_frame[np.newaxis, :], self.sequence_length - len(sequence), axis=0)
            sequence = np.concatenate([sequence, padding])
        
        return torch.FloatTensor(sequence), label

def prepare_data(db: GestureDatabase, test_size: float = 0.2, val_size: float = 0.1):
    """Preparar datos para entrenamiento"""
    # Obtener todos los gestos
    gestures = db.get_all_gestures()
    if not gestures:
        raise ValueError("No hay gestos en la base de datos")
    
    # Preparar datos
    sequences = []
    labels = []
    gesture_to_idx = {gesture["id"]: idx for idx, gesture in enumerate(gestures)}
    
    for gesture in gestures:
        gesture_sequences = db.get_gesture_sequences(gesture["id"])
        for sequence, _ in gesture_sequences:
            sequences.append(sequence)
            labels.append(gesture_to_idx[gesture["id"]])
    
    if not sequences:
        raise ValueError("No hay secuencias de gestos para entrenar")
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), gesture_to_idx

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: str
):
    """Entrenar el modelo"""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100. * correct / total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        logger.info(f'Val Accuracy: {accuracy:.2f}%')
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy
            }, save_path)
            logger.info(f'Modelo guardado en {save_path}')
    
    return train_losses, val_losses

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    gesture_names: List[str]
):
    """Evaluar el modelo en el conjunto de prueba"""
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gesture_names,
                yticklabels=gesture_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return test_loss, accuracy, cm

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo de reconocimiento de gestos LSC")
    parser.add_argument("--db_path", default="data/gestures.db", help="Ruta a la base de datos")
    parser.add_argument("--model_save_path", default="models/gesture_model.pth", help="Ruta para guardar el modelo")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--num_epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Tasa de aprendizaje")
    args = parser.parse_args()
    
    # Crear directorio para el modelo si no existe
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar base de datos
    db = GestureDatabase(args.db_path)
    
    try:
        # Preparar datos
        (X_train, y_train), (X_val, y_val), (X_test, y_test), gesture_to_idx = prepare_data(db)
        
        # Crear datasets
        train_dataset = GestureDataset(X_train, y_train)
        val_dataset = GestureDataset(X_val, y_val)
        test_dataset = GestureDataset(X_test, y_test)
        
        # Crear dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Crear modelo
        num_classes = len(gesture_to_idx)
        model = GestureModel(num_classes=num_classes).to(device)
        
        # Configurar entrenamiento
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            args.num_epochs, device, args.model_save_path
        )
        
        # Cargar mejor modelo
        checkpoint = torch.load(args.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluar modelo
        logger.info("Evaluando modelo...")
        gesture_names = [gesture["name"] for gesture in db.get_all_gestures()]
        test_loss, accuracy, cm = evaluate_model(model, test_loader, criterion, device, gesture_names)
        
        logger.info(f"Resultados finales:")
        logger.info(f"Pérdida en prueba: {test_loss:.4f}")
        logger.info(f"Precisión en prueba: {accuracy:.2f}%")
        logger.info(f"Matriz de confusión guardada en confusion_matrix.png")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main() 