import torch
import torch.nn as nn
from fastai.vision import *
from fastai.utils.mem import *
import numpy as np
import urllib.request
from pathlib import Path
import torchvision.transforms as T
from PIL import Image as PILImage
import requests
from io import BytesIO

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

class ArtLineModel:
    def __init__(self):
        self.model = None
        self.model_path = "ArtLine_650.pkl"
        
    def download_model(self):
        """Télécharge le modèle ArtLine si nécessaire"""
        if not Path(self.model_path).exists():
            print("Téléchargement du modèle ArtLine...")
            MODEL_URL = "https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1"
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Modèle téléchargé avec succès!")
    
    def load_model(self):
        """Charge le modèle ArtLine"""
        self.download_model()
        print("Chargement du modèle...")
        path = Path(".")
        self.model = load_learner(path, self.model_path)
        print("Modèle chargé avec succès!")
    
    def convert_image(self, pil_image):
        """Convertit une image PIL en dessin"""
        if self.model is None:
            self.load_model()
        
        # Convertir l'image PIL en format FastAI
        img_t = T.ToTensor()(pil_image)
        img_fast = Image(img_t)
        
        # Prédiction
        p, img_hr, b = self.model.predict(img_fast)
        
        # Convertir le résultat en image PIL
        img_array = image2np(img_hr)
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convertir de (C, H, W) à (H, W, C)
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        
        return PILImage.fromarray(img_array)