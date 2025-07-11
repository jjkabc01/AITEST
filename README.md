# Convertisseur Vidéo en Dessin Animé

Cette application utilise le modèle ArtLine pour convertir n'importe quelle vidéo en dessin animé. Elle extrait chaque frame de la vidéo, la convertit en dessin avec l'IA, puis recrée une nouvelle vidéo avec toutes les frames converties.

## Fonctionnalités

- ✨ Interface graphique intuitive
- 🎬 Support de multiples formats vidéo (MP4, AVI, MOV, MKV, WMV, FLV)
- 🎨 Conversion automatique en style dessin/line art
- 📊 Barre de progression et journal détaillé
- 💾 Sauvegarde automatique dans le dossier choisi
- 🔄 Traitement en arrière-plan (interface non bloquée)

## Installation

1. **Cloner le projet ArtLine original** (si pas déjà fait):
```bash
git clone https://github.com/vijishmadhavan/ArtLine.git
cd ArtLine
```

2. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

## Utilisation

### Interface Graphique

Lancez l'application avec l'interface graphique:
```bash
python main.py
```

1. Cliquez sur "Parcourir" pour sélectionner votre fichier vidéo
2. Choisissez le dossier de sortie (par défaut: dossier courant)
3. Cliquez sur "Convertir en Dessin Animé"
4. Attendez la fin du traitement (peut prendre du temps selon la longueur de la vidéo)
5. La vidéo convertie sera sauvegardée avec le suffixe "_cartoon"

### Utilisation Programmatique

```python
from artline_model import ArtLineModel
from video_processor import VideoProcessor

# Initialiser les modèles
artline = ArtLineModel()
processor = VideoProcessor(artline)

# Convertir une vidéo
processor.process_video("input_video.mp4", "output_cartoon.mp4")
```

## Comment ça marche

1. **Extraction des frames**: La vidéo est décomposée en images individuelles
2. **Conversion IA**: Chaque image est traitée par le modèle ArtLine pour créer un dessin
3. **Reconstruction**: Toutes les images converties sont assemblées en nouvelle vidéo

## Formats supportés

- **Entrée**: MP4, AVI, MOV, MKV, WMV, FLV
- **Sortie**: MP4

## Configuration requise

- Python 3.6+
- FastAI 1.0.61
- PyTorch 1.6.0
- OpenCV
- Pillow
- Au moins 4GB de RAM recommandés
- GPU recommandé pour de meilleures performances

## Limitations

- Le traitement peut être long pour les vidéos longues
- La qualité dépend de la qualité de la vidéo source
- Fonctionne mieux avec des visages et portraits
- Nécessite une connexion internet pour télécharger le modèle (première utilisation)

## Dépannage

### Erreur de mémoire
- Réduisez la résolution de votre vidéo source
- Fermez les autres applications

### Modèle non trouvé
- Vérifiez votre connexion internet
- Le modèle (650MB) se télécharge automatiquement au premier lancement

### Erreur de format vidéo
- Convertissez votre vidéo en MP4 avec un autre outil si nécessaire

## Crédits

- Modèle ArtLine: [vijishmadhavan/ArtLine](https://github.com/vijishmadhavan/ArtLine)
- Interface développée pour faciliter l'utilisation du modèle sur des vidéos

## Licence

Ce projet utilise le modèle ArtLine sous licence MIT.