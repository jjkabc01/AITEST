import os
import sys

def create_gui():
    """Crée une interface graphique simple pour sélectionner un fichier vidéo"""
    
    print("=== CONVERTISSEUR VIDEO EN DESSIN ANIME ===")
    print()
    
    # Demander le fichier vidéo d'entrée
    while True:
        print("Entrez le chemin vers votre fichier vidéo:")
        print("(ou tapez 'quit' pour quitter)")
        video_path = input("> ").strip()
        
        if video_path.lower() == 'quit':
            print("Au revoir!")
            return None, None
            
        if not video_path:
            print("Veuillez entrer un chemin de fichier.")
            continue
            
        if not os.path.exists(video_path):
            print(f"Erreur: Le fichier '{video_path}' n'existe pas.")
            continue
            
        # Vérifier l'extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in valid_extensions:
            print(f"Erreur: Format non supporté. Extensions valides: {', '.join(valid_extensions)}")
            continue
            
        break
    
    # Demander le dossier de sortie
    while True:
        print("\nEntrez le dossier de sortie (ou appuyez sur Entrée pour utiliser le dossier courant):")
        output_dir = input("> ").strip()
        
        if not output_dir:
            output_dir = os.getcwd()
            
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Dossier créé: {output_dir}")
            except Exception as e:
                print(f"Erreur lors de la création du dossier: {e}")
                continue
                
        break
    
    print(f"\nFichier sélectionné: {video_path}")
    print(f"Dossier de sortie: {output_dir}")
    print("\nAppuyez sur Entrée pour commencer la conversion...")
    input()
    
    return video_path, output_dir

def show_progress(current, total, message=""):
    """Affiche une barre de progression simple"""
    if total == 0:
        return
        
    percent = int((current / total) * 100)
    bar_length = 50
    filled_length = int(bar_length * current // total)
    
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r|{bar}| {percent}% {message}', end='', flush=True)
    
    if current == total:
        print()  # Nouvelle ligne à la fin

def show_completion(output_file):
    """Affiche le message de fin"""
    print("\n" + "="*60)
    print("🎉 CONVERSION TERMINÉE AVEC SUCCÈS! 🎉")
    print("="*60)
    print(f"Fichier de sortie: {output_file}")
    print("\nVotre vidéo a été convertie en dessin animé!")
    print("="*60)

if __name__ == "__main__":
    video_path, output_dir = create_gui()
    if video_path and output_dir:
        print(f"Traitement de: {video_path}")
        print(f"Sortie dans: {output_dir}")