
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
                "¡Todas las librerías están funcionando correctamente!\n"
                "El sistema está listo para usar."
            )
        except ImportError as e:
            messagebox.showerror(
                "Error", 
                f"Error en las importaciones:\n{e}"
            )

if __name__ == "__main__":
    app = PruebaApp()
    app.mainloop()
