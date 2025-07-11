#!/usr/bin/env python3
"""
Convertisseur Vidéo en Dessin Animé
Utilise le modèle ArtLine pour convertir chaque frame d'une vidéo en dessin
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Point d'entrée principal de l'application"""
    try:
        # Importer et lancer l'interface graphique
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Erreur d'importation: {e}")
        print("Assurez-vous que toutes les dépendances sont installées:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du lancement de l'application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()