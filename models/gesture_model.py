import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

class GestureModel(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 63, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.5):
        """
        Modelo para reconocimiento de gestos LSC
        
        Args:
            num_classes: Número de clases de gestos
            input_size: Tamaño de entrada (21 landmarks * 3 coordenadas)
            hidden_size: Tamaño de las capas ocultas
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        
        # Capa de normalización
        self.normalize = nn.BatchNorm1d(input_size)
        
        # Capa LSTM para procesar secuencias
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Mecanismo de atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializar pesos de manera óptima"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Tensor de entrada (batch_size, sequence_length, input_size)
            
        Returns:
            Tuple de (logits, attention_weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # Normalizar entrada
        x = self.normalize(x.transpose(1, 2)).transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # Atención
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size * 2)
        
        # Clasificación
        logits = self.fc(context)
        
        return logits, attention_weights
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realizar predicción
        
        Args:
            x: Tensor de entrada (batch_size, sequence_length, input_size)
            
        Returns:
            Tuple de (predicciones, probabilidades)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtener pesos de atención para visualización
        
        Args:
            x: Tensor de entrada (batch_size, sequence_length, input_size)
            
        Returns:
            Pesos de atención (batch_size, sequence_length)
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self(x)
            attention_weights = attention_weights.squeeze(-1)
        return attention_weights

class GestureRecognizer:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureModel(num_classes=100).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # Configuración de procesamiento
        self.sequence_length = 30
        self.input_buffer = []
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> torch.Tensor:
        # Normalizar landmarks
        landmarks = landmarks.reshape(-1)
        landmarks = (landmarks - landmarks.mean()) / landmarks.std()
        
        # Convertir a tensor
        return torch.FloatTensor(landmarks).unsqueeze(0)
    
    def process_sequence(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        if len(landmarks_sequence) < self.sequence_length:
            return {"gesture": "unknown", "confidence": 0.0}
        
        # Preprocesar secuencia
        sequence = torch.stack([
            self.preprocess_landmarks(landmarks)
            for landmarks in landmarks_sequence[-self.sequence_length:]
        ]).unsqueeze(0)
        
        # Mover a dispositivo
        sequence = sequence.to(self.device)
        
        # Inferencia
        with torch.no_grad():
            logits, attention = self.model(sequence)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        return {
            "gesture": self.idx_to_gesture[prediction.item()],
            "confidence": confidence.item()
        }
    
    def update_buffer(self, landmarks: np.ndarray):
        self.input_buffer.append(landmarks)
        if len(self.input_buffer) > self.sequence_length:
            self.input_buffer.pop(0)
    
    def get_prediction(self) -> Dict:
        if len(self.input_buffer) < self.sequence_length:
            return {"gesture": "unknown", "confidence": 0.0}
        
        return self.process_sequence(self.input_buffer) 