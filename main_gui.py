#!/usr/bin/env python3
"""
Application principale avec interface graphique pour convertir une vidéo en dessin animé
"""

import os
import sys
from gui_simple import create_gui, show_progress, show_completion
from video_processor_simple import VideoProcessor
from simple_artline import SimpleArtLine

def main():
    """Fonction principale avec interface graphique"""
    
    print("Initialisation de l'application...")
    
    # Interface graphique pour sélectionner les fichiers
    video_path, output_dir = create_gui()
    
    if not video_path or not output_dir:
        return
    
    try:
        # Initialiser les composants
        print("\n📋 Initialisation des composants...")
        artline = SimpleArtLine()
        processor = VideoProcessor()
        
        # Générer le nom du fichier de sortie
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{video_name}_cartoon.mp4")
        
        print(f"🎬 Début du traitement de: {os.path.basename(video_path)}")
        print(f"📁 Fichier de sortie: {os.path.basename(output_file)}")
        
        # Callback pour la progression
        def progress_callback(current, total, stage):
            message = f"({stage}) {current}/{total}"
            show_progress(current, total, message)
        
        # Traiter la vidéo
        success = processor.process_video(
            video_path, 
            output_file, 
            artline, 
            progress_callback
        )
        
        if success:
            show_completion(output_file)
        else:
            print("\n❌ Erreur lors du traitement de la vidéo")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Traitement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()