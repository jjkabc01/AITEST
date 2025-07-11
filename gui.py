import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from pathlib import Path
from artline_model import ArtLineModel
from video_processor import VideoProcessor

class VideoToCartoonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Convertisseur Vidéo en Dessin Animé")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Variables
        self.input_video_path = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.getcwd())
        self.processing = False
        
        # Initialiser les modèles
        self.artline_model = ArtLineModel()
        self.video_processor = VideoProcessor(self.artline_model)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Titre
        title_label = ttk.Label(main_frame, text="Convertisseur Vidéo en Dessin Animé", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Sélection du fichier vidéo
        ttk.Label(main_frame, text="Fichier vidéo source:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        
        self.video_entry = ttk.Entry(video_frame, textvariable=self.input_video_path, width=50)
        self.video_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(video_frame, text="Parcourir", 
                  command=self.select_video_file).grid(row=0, column=1)
        
        # Dossier de sortie
        ttk.Label(main_frame, text="Dossier de sortie:").grid(row=3, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_folder, width=50)
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(output_frame, text="Parcourir", 
                  command=self.select_output_folder).grid(row=0, column=1)
        
        # Bouton de conversion
        self.convert_button = ttk.Button(main_frame, text="Convertir en Dessin Animé", 
                                        command=self.start_conversion, style='Accent.TButton')
        self.convert_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Barre de progression
        self.progress_var = tk.StringVar(value="Prêt à convertir")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=6, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal", padding="10")
        log_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=20)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configuration du redimensionnement
        main_frame.rowconfigure(8, weight=1)
    
    def log_message(self, message):
        """Ajoute un message au journal"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def select_video_file(self):
        """Sélectionne le fichier vidéo source"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[
                ("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if file_path:
            self.input_video_path.set(file_path)
    
    def select_output_folder(self):
        """Sélectionne le dossier de sortie"""
        folder_path = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder_path:
            self.output_folder.set(folder_path)
    
    def start_conversion(self):
        """Démarre la conversion dans un thread séparé"""
        if self.processing:
            return
        
        if not self.input_video_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier vidéo")
            return
        
        if not os.path.exists(self.input_video_path.get()):
            messagebox.showerror("Erreur", "Le fichier vidéo sélectionné n'existe pas")
            return
        
        self.processing = True
        self.convert_button.config(state='disabled')
        self.progress_bar.start()
        self.progress_var.set("Conversion en cours...")
        self.log_text.delete(1.0, tk.END)
        
        # Lancer la conversion dans un thread séparé
        thread = threading.Thread(target=self.convert_video)
        thread.daemon = True
        thread.start()
    
    def convert_video(self):
        """Effectue la conversion vidéo"""
        try:
            input_path = self.input_video_path.get()
            output_dir = self.output_folder.get()
            
            # Générer le nom du fichier de sortie
            input_name = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_cartoon.mp4")
            
            self.log_message(f"Début de la conversion: {input_path}")
            self.log_message(f"Fichier de sortie: {output_path}")
            
            # Rediriger les prints vers le log
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Effectuer la conversion
                self.video_processor.process_video(input_path, output_path)
                
                # Récupérer les messages
                output = sys.stdout.getvalue()
                for line in output.split('\n'):
                    if line.strip():
                        self.root.after(0, self.log_message, line.strip())
                
            finally:
                sys.stdout = old_stdout
            
            # Succès
            self.root.after(0, self.conversion_complete, output_path)
            
        except Exception as e:
            error_msg = f"Erreur lors de la conversion: {str(e)}"
            self.root.after(0, self.conversion_error, error_msg)
    
    def conversion_complete(self, output_path):
        """Appelé quand la conversion est terminée"""
        self.processing = False
        self.convert_button.config(state='normal')
        self.progress_bar.stop()
        self.progress_var.set("Conversion terminée avec succès!")
        
        self.log_message("=" * 50)
        self.log_message("CONVERSION TERMINÉE AVEC SUCCÈS!")
        self.log_message(f"Fichier créé: {output_path}")
        self.log_message("=" * 50)
        
        messagebox.showinfo("Succès", f"Conversion terminée!\nFichier créé: {output_path}")
    
    def conversion_error(self, error_msg):
        """Appelé en cas d'erreur"""
        self.processing = False
        self.convert_button.config(state='normal')
        self.progress_bar.stop()
        self.progress_var.set("Erreur lors de la conversion")
        
        self.log_message("=" * 50)
        self.log_message("ERREUR LORS DE LA CONVERSION!")
        self.log_message(error_msg)
        self.log_message("=" * 50)
        
        messagebox.showerror("Erreur", error_msg)

def main():
    root = tk.Tk()
    app = VideoToCartoonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()