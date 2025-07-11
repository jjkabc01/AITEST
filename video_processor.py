import cv2
import os
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm
import tempfile
import shutil

class VideoProcessor:
    def __init__(self, artline_model):
        self.artline_model = artline_model
        
    def extract_frames(self, video_path, temp_dir):
        """Extrait toutes les frames d'une vidéo"""
        print("Extraction des frames de la vidéo...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Extraction des frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convertir BGR (OpenCV) en RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(frame_rgb)
                
                # Sauvegarder la frame temporairement
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                pil_image.save(frame_path)
                frames.append(frame_path)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        print(f"Extraction terminée: {len(frames)} frames extraites")
        return frames, fps
    
    def convert_frames(self, frame_paths, output_dir):
        """Convertit toutes les frames en dessins"""
        print("Conversion des frames en dessins...")
        
        converted_frames = []
        
        with tqdm(total=len(frame_paths), desc="Conversion en dessins") as pbar:
            for i, frame_path in enumerate(frame_paths):
                # Charger l'image
                pil_image = PILImage.open(frame_path).convert("RGB")
                
                # Convertir en dessin
                cartoon_image = self.artline_model.convert_image(pil_image)
                
                # Sauvegarder l'image convertie
                output_path = os.path.join(output_dir, f"cartoon_{i:06d}.jpg")
                cartoon_image.save(output_path)
                converted_frames.append(output_path)
                
                pbar.update(1)
        
        print(f"Conversion terminée: {len(converted_frames)} frames converties")
        return converted_frames
    
    def create_video(self, frame_paths, output_path, fps):
        """Crée une vidéo à partir des frames converties"""
        print("Création de la vidéo finale...")
        
        if not frame_paths:
            raise ValueError("Aucune frame à traiter")
        
        # Lire la première image pour obtenir les dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width, layers = first_frame.shape
        
        # Créer le writer vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        with tqdm(total=len(frame_paths), desc="Création de la vidéo") as pbar:
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
                pbar.update(1)
        
        video_writer.release()
        print(f"Vidéo créée avec succès: {output_path}")
    
    def process_video(self, input_video_path, output_video_path):
        """Traite une vidéo complète: extraction -> conversion -> reconstruction"""
        print(f"Début du traitement de la vidéo: {input_video_path}")
        
        # Créer des dossiers temporaires
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, "frames")
            cartoon_dir = os.path.join(temp_dir, "cartoon")
            os.makedirs(frames_dir)
            os.makedirs(cartoon_dir)
            
            try:
                # Étape 1: Extraire les frames
                frame_paths, fps = self.extract_frames(input_video_path, frames_dir)
                
                # Étape 2: Convertir les frames
                cartoon_paths = self.convert_frames(frame_paths, cartoon_dir)
                
                # Étape 3: Créer la vidéo finale
                self.create_video(cartoon_paths, output_video_path, fps)
                
                print("Traitement terminé avec succès!")
                
            except Exception as e:
                print(f"Erreur lors du traitement: {str(e)}")
                raise