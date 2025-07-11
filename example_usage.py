#!/usr/bin/env python3
"""
Exemple d'utilisation du convertisseur vidéo en dessin animé
"""

import os
from artline_model import ArtLineModel
from video_processor import VideoProcessor

def example_convert_video():
    """Exemple de conversion d'une vidéo"""
    
    # Chemins des fichiers
    input_video = "example_video.mp4"  # Remplacez par votre fichier
    output_video = "example_cartoon.mp4"
    
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(input_video):
        print(f"Erreur: Le fichier {input_video} n'existe pas")
        print("Veuillez remplacer 'example_video.mp4' par le chemin de votre vidéo")
        return
    
    try:
        # Initialiser les modèles
        print("Initialisation du modèle ArtLine...")
        artline_model = ArtLineModel()
        video_processor = VideoProcessor(artline_model)
        
        # Convertir la vidéo
        print(f"Conversion de {input_video} vers {output_video}")
        video_processor.process_video(input_video, output_video)
        
        print("Conversion terminée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de la conversion: {e}")

def example_convert_single_image():
    """Exemple de conversion d'une seule image"""
    
    from PIL import Image
    
    # Charger une image d'exemple depuis une URL
    import requests
    from io import BytesIO
    
    try:
        # Télécharger une image d'exemple
        url = 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Initialiser le modèle
        artline_model = ArtLineModel()
        
        # Convertir l'image
        cartoon_img = artline_model.convert_image(img)
        
        # Sauvegarder le résultat
        cartoon_img.save("example_cartoon_image.jpg")
        print("Image convertie sauvegardée: example_cartoon_image.jpg")
        
    except Exception as e:
        print(f"Erreur lors de la conversion d'image: {e}")

if __name__ == "__main__":
    print("=== Exemples d'utilisation du Convertisseur Vidéo en Dessin Animé ===\n")
    
    print("1. Conversion d'une image unique:")
    example_convert_single_image()
    
    print("\n2. Conversion d'une vidéo:")
    print("   (Décommentez la ligne suivante et ajoutez votre fichier vidéo)")
    # example_convert_video()
    
    print("\nPour utiliser l'interface graphique, lancez: python main.py")