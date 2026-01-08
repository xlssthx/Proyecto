"""
Sistema de Seguridad SIVISEC - Punto de entrada principal
"""
import tkinter as tk
from controlador import SistemaSeguridadControlador


def main():
    """Función principal del programa"""
    # Crear ventana principal
    root = tk.Tk()
    
    # Crear la aplicación
    app = SistemaSeguridadControlador(root)
    
    # Iniciar loop principal
    root.mainloop()


if __name__ == "__main__":
    main()
    