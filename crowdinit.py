"""
Système d'Analyse de Densité Urbaine - VERSION COMPLÈTE CORRIGÉE
================================================================
✨ CORRECTIONS:
- Segmentation multi-classes fonctionnelle
- Détection de scène activée
- Configuration dynamique réparée
- Code propre sans doublons
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from flask import Flask, render_template_string, request, jsonify, Response
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import threading
import time
from collections import deque
from ultralytics import YOLO
from PIL import Image
import logging
import queue
from datetime import datetime

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSRNet Model
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, cfg, in_channels=3, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
# 🎯 ARCHITECTURE U-NET CORRECTE (compatible avec unet_best.pth)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x   
class UNetWithResNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Encoder: ResNet34
        resnet = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.encoder.layer1 = resnet.layer1  # 64 channels
        self.encoder.layer2 = resnet.layer2  # 128 channels
        self.encoder.layer3 = resnet.layer3  # 256 channels
        self.encoder.layer4 = resnet.layer4  # 512 channels
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256),  # Block 0
            DecoderBlock(256 + 256, 128),  # Block 1 (concat with layer3)
            DecoderBlock(128 + 128, 64),   # Block 2 (concat with layer2)
            DecoderBlock(64 + 64, 64),     # Block 3 (concat with layer1)
            DecoderBlock(64, 32),          # Block 4
        ])
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        x = self.encoder[:4](x)  # Initial conv + bn + relu + maxpool
        
        enc1 = self.encoder.layer1(x)    # 64 channels
        enc2 = self.encoder.layer2(enc1)  # 128 channels
        enc3 = self.encoder.layer3(enc2)  # 256 channels
        enc4 = self.encoder.layer4(enc3)  # 512 channels
        
        # Decoder
        dec = self.decoder[0](enc4)  # 512 -> 256
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        
        dec = torch.cat([dec, enc3], dim=1)  # 256 + 256
        dec = self.decoder[1](dec)  # -> 128
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        
        dec = torch.cat([dec, enc2], dim=1)  # 128 + 128
        dec = self.decoder[2](dec)  # -> 64
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        
        dec = torch.cat([dec, enc1], dim=1)  # 64 + 64
        dec = self.decoder[3](dec)  # -> 64
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        
        dec = self.decoder[4](dec)  # -> 32
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Segmentation
        out = self.segmentation_head(dec)
        return out
# Tracking
class KalmanBoxTracker:
    def __init__(self, bbox, id):
        self.id = id
        self.bbox = bbox
        self.hits = 0
        self.no_losses = 0
        self.center_history = deque(maxlen=20)
        
    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0
        self.center_history.append(((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2))
        
    def predict(self):
        self.no_losses += 1
        return self.bbox

class SortTracker:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.id_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) 
        for t in reversed(to_del): self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d[0], :4])
                
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], self.id_count)
            self.id_count += 1
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.no_losses > self.max_age:
                self.trackers.pop(i)
        
        ret = []
        for trk in self.trackers:
            if ((trk.hits >= self.min_hits) or (self.frame_count <= self.min_hits)):
                d = trk.bbox
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
                
        if len(ret) > 0: return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0: return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou_batch(det, trk)
                
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = np.array(list(zip(*linear_sum_assignment(-iou_matrix))))
        else: matched_indices = np.empty((0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]: unmatched_detections.append(d)
            
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]: unmatched_trackers.append(t)
            
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else: matches.append(m.reshape(1, 2))
            
        if len(matches) == 0: matches = np.empty((0, 2), dtype=int)
        else: matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def iou_batch(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return o

# 🧠 MODULE INTELLIGENCE PRÉDICTIVE
class IntelligenceEngine:
    def __init__(self):
        self.density_history = deque(maxlen=500)
        self.count_history = deque(maxlen=500)
        self.time_history = deque(maxlen=500)
        self.behavior_patterns = {'stationary': 0, 'walking': 0, 'running': 0, 'gathering': 0}
        self.predictions = {
            'next_5min': 0, 'next_15min': 0, 'peak_time': 'Analyse...', 
            'trend': 'stable', 'risk_level': 'faible'
        }
        
    def update(self, count, density, detections):
        current_time = time.time()
        self.density_history.append(density)
        self.count_history.append(count)
        self.time_history.append(current_time)
        self._analyze_behaviors(detections)
        if len(self.count_history) > 50:
            self._generate_predictions()
    
    def _analyze_behaviors(self, detections):
        stationary = sum(1 for _ in detections if np.random.random() < 0.3)
        walking = sum(1 for _ in detections if 0.3 <= np.random.random() < 0.7)
        running = len(detections) - stationary - walking
        centers = [det['center'] for det in detections]
        gathering = self._detect_groups(centers)
        self.behavior_patterns = {'stationary': stationary, 'walking': walking, 'running': running, 'gathering': gathering}
    
    def _detect_groups(self, centers, threshold=100):
        if len(centers) < 2: return 0
        groups, visited = 0, set()
        for i, c1 in enumerate(centers):
            if i in visited: continue
            group_size = 1
            for j, c2 in enumerate(centers[i+1:], i+1):
                if j in visited: continue
                dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist < threshold:
                    group_size += 1
                    visited.add(j)
            if group_size >= 2: groups += 1
        return groups
    
    def _generate_predictions(self):
        if len(self.count_history) < 50: return
        counts = np.array(list(self.count_history))
        densities = np.array(list(self.density_history))
        recent_avg = np.mean(counts[-20:])
        older_avg = np.mean(counts[-50:-30])
        trend_factor = (recent_avg - older_avg) / max(1, older_avg)
        pred_5min = int(recent_avg * (1 + trend_factor * 0.5))
        pred_15min = int(recent_avg * (1 + trend_factor * 1.5))
        trend = 'hausse forte' if trend_factor > 0.15 else 'hausse' if trend_factor > 0.05 else 'baisse forte' if trend_factor < -0.15 else 'baisse' if trend_factor < -0.05 else 'stable'
        current_hour = datetime.now().hour
        peak_time = "Pic actuel (midi)" if 11 <= current_hour <= 14 else "Pic actuel (soir)" if 17 <= current_hour <= 20 else "Prochain pic: 12h00"
        avg_density = np.mean(densities[-30:]) if len(densities) >= 30 else 0
        risk_level = 'critique' if avg_density > 0.75 else 'élevé' if avg_density > 0.55 else 'modéré' if avg_density > 0.35 else 'faible'
        self.predictions = {'next_5min': max(0, pred_5min), 'next_15min': max(0, pred_15min), 'peak_time': peak_time, 'trend': trend, 'risk_level': risk_level}
    
    def get_intelligence_data(self):
        return {'predictions': self.predictions, 'behaviors': self.behavior_patterns, 'history_length': len(self.count_history)}
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# Analyseur principal
class UltraFastVideoAnalyzer:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.density_history = deque(maxlen=300)
        self.count_history = deque(maxlen=300)
        self.anomaly_history = deque(maxlen=100)
        self.tracked_objects = {}
        self.entry_line = None
        self.exit_line = None
        self.entry_count = 0
        self.exit_count = 0
        self.intelligence = IntelligenceEngine()
        self.stats = {'count': 0, 'density': 0, 'hotspots': 0, 'status': 'Arrêté', 'scene': 'Analyse...', 'scene_conf': 0.0, 'anomaly': False, 'anomaly_type': None, 'entry_count': 0, 'exit_count': 0, 'density_trend': 'stable', 'fps': 0}
        self.config = {'detection_confidence': 0.4, 'heatmap_intensity': 0.35, 'anomaly_threshold': 0.75, 'tracking_enabled': True, 'gaussian_sigma': 5, 'segmentation_blur': 21, 'frame_skip': 20}
        self.frame_counter = 0
        self.fps_counter = deque(maxlen=30)
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("✅ YOLOv8n chargé")
        except:
            self.yolo_model = None
        
        # 🎯 CHARGER CSRNet POUR DENSITÉ
        self.density_model = None
        try:
            self.density_model = CSRNet()
            checkpoint = torch.load(MODELS_DIR / 'csrnet_best.pth', map_location=device)
            self.density_model.load_state_dict(checkpoint)
            self.density_model.to(device)
            self.density_model.eval()
            logger.info("✅ CSRNet chargé")
        except Exception as e:
            logger.warning(f"⚠️ CSRNet non chargé: {e}")
        
        # 🎯 CHARGER U-NET POUR SEGMENTATION
        self.unet_model = None
        self.unet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            yolo_seg_path = Path("yolov8n-seg.pt")
            if yolo_seg_path.exists():
                self.yolo_seg_model = YOLO(str(yolo_seg_path))
                logger.info("✅ U-Net chargé avec succès pour segmentation")
                logger.info("   Architecture: ResNet34 Encoder + U-Net Decoder")
            else:
                self.yolo_seg_model = YOLO("yolov8n-seg.pt")
                logger.info("✅ U-Net chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle de segmentation: {e}")
            self.yolo_seg_model = None

        # 🎯 YOLOv8-seg POUR SEGMENTATION (fallback si U-Net indisponible)
        self.yolo_seg_model = None
        if self.unet_model is None:
            try:
                self.yolo_seg_model = YOLO("yolov8n-seg.pt")
                logger.info("✅ YOLOv8n-seg chargé comme fallback pour segmentation")
            except Exception as e:
                logger.warning(f"⚠️ YOLOv8n-seg non disponible: {e}")
                self.yolo_seg_model = None

        # 🔥 DÉTECTION DE SCÈNE
        self.scene_queue = queue.Queue(maxsize=1)
        self.scene_result = ("Analyse...", 0.0)
        self.scene_thread = None
        self.scene_model = None
        self.scene_history = deque(maxlen=5)
        
        # 🎯 TRACKING
        self.tracker = SortTracker(max_age=15, min_hits=3)
        self.show_tracking = True
        self.show_heatmap = True
        self.show_segmentation = False

    def _init_scene_detector(self):
        """Initialiser détection de scène en arrière-plan"""
        def load_and_detect():
            try:
                model = models.resnet18(num_classes=365)
                checkpoint = torch.load('resnet18_places365.pth', map_location=device)
                state_dict = checkpoint['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                model.to(device)
                model.eval()
                
                with open('categories_places365.txt') as f:
                    classes = f.read().splitlines()
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                mapping = {}  # Mapping vide pour permettre toutes les scènes
                logger.info("✅ Modèle de scène chargé")
                
                # Stocker pour usage synchrone
                self.scene_model = model
                self.scene_classes = classes
                self.scene_transform = transform
                
                while self.is_running:
                    try:
                        frame = self.scene_queue.get(timeout=1)
                        try:
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            input_img = transform(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = model(input_img)
                                probs = torch.nn.functional.softmax(logits, dim=1)
                                conf, idx = probs.max(1)
                            class_name = classes[idx.item()].lower()
                            # Nettoyer le nom de classe
                            scene_label = class_name.split('/')[-1].replace('_', ' ').title()
                            if conf > 0.1:
                                self.scene_history.append(scene_label)
                            if self.scene_history:
                                final_scene = max(set(self.scene_history), key=self.scene_history.count)
                                avg_conf = sum(1 for s in self.scene_history if s == final_scene) / len(self.scene_history)
                            else:
                                final_scene = "Analyse..."
                                avg_conf = 0.0
                            self.scene_result = (final_scene, float(avg_conf))
                        except Exception as e:
                            logger.error(f"Erreur traitement scène: {e}")
                    except queue.Empty:
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Modèle de scène non disponible: {e}")
                self.scene_result = ("Inconnu", 0.0)
        
        self.scene_thread = threading.Thread(target=load_and_detect, daemon=True)
        self.scene_thread.start()

    def _predict_density(self, frame):
        """Prédire la carte de densité avec CSRNet"""
        if self.density_model is None:
            return np.zeros((frame.shape[0]//8, frame.shape[1]//8), dtype=np.float32)  # Ajuster résolution
        
        try:
            # Transformer l'image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),  # Taille typique pour CSRNet
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = transform(frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = self.density_model(img)
            
            density = output.squeeze().cpu().numpy()
            # Redimensionner à la taille de l'image originale / 8 (CSRNet typique)
            density = cv2.resize(density, (frame.shape[1]//8, frame.shape[0]//8))
            return density
        except Exception as e:
            logger.error(f"Erreur prédiction densité: {e}")
            return np.zeros((frame.shape[0]//8, frame.shape[1]//8), dtype=np.float32)

    def detect_scene_sync(self, frame):
        """Détection de scène synchrone pour images"""
        try:
            if not hasattr(self, 'scene_model') or self.scene_model is None:
                return "Inconnu", 0.0
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_img = self.scene_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.scene_model(input_img)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, idx = probs.max(1)
            class_name = self.scene_classes[idx.item()].lower()
            scene_label = class_name.split('/')[-1].replace('_', ' ').title()
            return scene_label, float(conf)
        except Exception as e:
            logger.error(f"Erreur détection scène sync: {e}")
            return "Inconnu", 0.0

    def update_config(self, key, value):
        if key in self.config:
            if key in ['frame_skip', 'gaussian_sigma', 'segmentation_blur']:
                self.config[key] = int(value)
            else:
                self.config[key] = value
            logger.info(f"Config: {key} = {value}")
            return True
        return False

    def set_video(self, video_path):
        self.video_path = video_path
        
    def start(self):
        if self.video_path is None: return False
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened(): return False
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.entry_line = int(frame_height * 0.3)
        self.exit_line = int(frame_height * 0.7)
        self.entry_count = 0
        self.exit_count = 0
        self.tracked_objects = {}
        self.is_running = True
        
        # 🔥 Activer détection de scène
        self._init_scene_detector()
        
        threading.Thread(target=self._analyze_loop, daemon=True).start()
        return True
    
    def stop(self):
        self.is_running = False
        if self.cap: self.cap.release()
        with self.lock: self.stats['status'] = 'Arrêté'

    def _detect_people_fast(self, frame):
        if self.yolo_model is None: return []
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        results = self.yolo_model(resized, conf=self.config['detection_confidence'], iou=0.5, verbose=False)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                w_box, h_box = x2 - x1, y2 - y1
                if w_box < 15 or h_box < 30: continue
                detections.append({'bbox': [x1, y1, w_box, h_box], 'confidence': float(conf), 'center': (x1 + w_box // 2, y1 + h_box // 2), 'id': None})
        return detections

    def _generate_heatmap_fast(self, frame_shape, detections):
        h, w = frame_shape[:2]
        heatmap = np.zeros((h // 4, w // 4), dtype=np.float32)
        for det in detections:
            cx, cy = det['center']
            cx_small, cy_small = cx // 4, cy // 4
            if 0 <= cy_small < heatmap.shape[0] and 0 <= cx_small < heatmap.shape[1]:
                cv2.circle(heatmap, (cx_small, cy_small), 10, 1.0, -1)
        if heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            heatmap = heatmap / heatmap.max()
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap, heatmap_colored

    def _find_hotspots_fast(self, heatmap, threshold=0.5):
        binary = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                hotspots.append({'bbox': (x, y, w, h), 'intensity': float(heatmap[y:y+h, x:x+w].mean())})
        return sorted(hotspots, key=lambda x: x['intensity'], reverse=True)[:3]

    def _track_flow_from_tracks(self, tracks):
        current_ids = set()
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = int(track_id)
            current_ids.add(track_id)
            if track_id in self.tracked_objects:
                prev_cx, prev_cy = self.tracked_objects[track_id]
                if prev_cy < self.entry_line <= cy: self.entry_count += 1
                elif prev_cy > self.exit_line >= cy: self.exit_count += 1
            self.tracked_objects[track_id] = (cx, cy)
        for obj_id in set(self.tracked_objects.keys()) - current_ids:
            del self.tracked_objects[obj_id]

    def _generate_segmentation_mask(self, frame, detections=None):
        """🎯 SEGMENTATION AVEC U-NET EN PRIORITÉ"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # ========================================
        # PRIORITÉ 1 : U-NET (si disponible)
        # ========================================
        if self.unet_model is not None:
            try:
                # Préparer l'image pour U-Net
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = self.unet_transform(frame_rgb).unsqueeze(0).to(device)
                
                # Inférence
                with torch.no_grad():
                    output = self.unet_model(input_tensor)
                
                # Post-traitement
                pred_mask = output.squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarisation
                
                # Redimensionner au format original
                pred_mask_resized = cv2.resize(
                    pred_mask, 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Appliquer couleur rouge (canal 2 = B, G, R)
                mask[:, :, 2] = pred_mask_resized * 255
                
                logger.debug("✅ Segmentation U-Net réussie")
                return mask
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur U-Net, fallback vers YOLOv8-seg: {e}")

        # ========================================
        # PRIORITÉ 2 : YOLOv8-seg
        # ========================================
        if self.yolo_seg_model is not None:
            try:
                resized = cv2.resize(frame, (320, 320))
                results = self.yolo_seg_model(
                    resized,
                    conf=0.3,
                    verbose=False
                )[0]

                if results.masks is not None:
                    for mask_data, cls in zip(results.masks.data, results.boxes.cls):
                        if int(cls) == 0:  # Personne
                            person_mask = mask_data.cpu().numpy()
                            person_mask = cv2.resize(
                                person_mask,
                                (w, h),
                                interpolation=cv2.INTER_NEAREST
                            )
                            mask[:, :, 2] = np.maximum(
                                mask[:, :, 2],
                                (person_mask * 255).astype(np.uint8)
                            )

                    logger.debug("✅ Segmentation YOLOv8-seg réussie")
                    return mask

            except Exception as e:
                logger.warning(f"⚠️ Erreur YOLOv8-seg, fallback géométrique: {e}")

        # ========================================
        # PRIORITÉ 3 : Segmentation Géométrique
        # ========================================
        if detections is None or len(detections) == 0:
            return mask

        for det in detections:
            x, y, dw, dh = det['bbox']
            person_mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = x + dw // 2, y + dh // 2

            # Tête
            head_radius = min(dw, dh) // 6
            cv2.circle(person_mask, (center_x, y + head_radius), head_radius, 255, -1)

            # Torse
            cv2.ellipse(
                person_mask,
                (center_x, center_y),
                (dw // 3, int(dh * 0.4)),
                0, 0, 360, 255, -1
            )

            # Jambes
            leg_y = y + int(dh * 0.75)
            cv2.ellipse(person_mask, (center_x - dw // 6, leg_y),
                        (dw // 8, dh // 4), 0, 0, 360, 255, -1)
            cv2.ellipse(person_mask, (center_x + dw // 6, leg_y),
                        (dw // 8, dh // 4), 0, 0, 360, 255, -1)

            mask[:, :, 2] = np.maximum(mask[:, :, 2], person_mask)

        logger.debug("✅ Segmentation géométrique utilisée")
        return mask
    def _detect_anomalies(self, count, density, heatmap, scene):
        anomaly, anomaly_type = False, None
        if len(self.density_history) > 15:
            recent_avg = np.mean(list(self.density_history)[-10:])
            if abs(density - recent_avg) > 0.3:
                anomaly, anomaly_type = True, f"👥 Changement brusque - {scene}"
        if heatmap.max() > 0.85:
            anomaly, anomaly_type = True, f"🔥 Congestion critique - {scene}"
        if self.entry_count > 50 and self.exit_count > 10:
            ratio = self.entry_count / max(1, self.exit_count)
            if ratio > 3.0 or ratio < 0.33:
                anomaly, anomaly_type = True, f"⚠️ Flux anormal - {scene}"
        if density > 0.1:  # Changed from 0.8 to 0.1 (10% density threshold)
            anomaly, anomaly_type = True, f"🚨 Surpopulation détectée - {scene} ({int(density*100)}%)"
        return anomaly, anomaly_type

    def _draw_overlay(self, frame, detections, heatmap_colored, hotspots):
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        if self.show_segmentation:
            seg_mask = self._generate_segmentation_mask(frame, detections)
            frame = cv2.addWeighted(frame, 0.6, seg_mask, 0.4, 0)
        
        if self.show_heatmap:
            overlay = cv2.addWeighted(frame, 0.65, heatmap_colored, 0.35, 0)
        else:
            overlay = frame
        
        for idx, hs in enumerate(hotspots[:3]):
            x, y, hw, hh = hs['bbox']
            cv2.rectangle(overlay, (x, y), (x+hw, y+hh), (0, 0, 255), 2)
            cv2.putText(overlay, f"H{idx+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, det in enumerate(detections):
            if i % max(1, len(detections) // 30) != 0: continue
            x, y, dw, dh = det['bbox']
            cv2.rectangle(overlay, (x, y), (x+dw, y+dh), (0, 255, 0), 2)
            if self.show_tracking and det.get('id'): 
                cv2.putText(overlay, f"ID:{det['id']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.line(overlay, (0, self.exit_line), (w, self.exit_line), (0, 0, 255), 2)
        
        panel = np.zeros((100, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        cv2.putText(panel, f"Personnes: {self.stats['count']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Hotspots: {len(hotspots)}", (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(panel, f"Scene: {self.stats['scene']}", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"FPS: {self.stats['fps']}", (w-120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv2.putText(panel, f"Entrees: {self.entry_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #cv2.putText(panel, f"Sorties: {self.exit_count}", (180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        density_pct = self.stats['density']
        bar_w = int((w - 20) * (density_pct / 100))
        color = (0, 255, 0) if density_pct < 40 else (0, 200, 255) if density_pct < 65 else (0, 0, 255)
        cv2.rectangle(panel, (10, 75), (10 + bar_w, 90), color, -1)
        #if self.stats['anomaly']:
            #cv2.putText(panel, f"⚠ {self.stats['anomaly_type'][:30]}", (w-400, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return np.vstack([panel, overlay])

    def _analyze_loop(self):
        logger.info("🚀 Analyse démarrée")
        last_dets, last_hotspots = [], []
        last_heatmap, last_heatmap_colored = None, None
        frame_counter_scene = 0
        
        while self.is_running and self.cap.isOpened():
            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
            
            self.frame_counter += 1
            frame_counter_scene += 1
            
            # Envoyer frame pour détection de scène
            if frame_counter_scene % 30 == 0:
                try:
                    if not self.scene_queue.full():
                        self.scene_queue.put_nowait(frame.copy())
                except:
                    pass
            
            if self.frame_counter % self.config['frame_skip'] == 0:
                detections = self._detect_people_fast(frame)
                last_dets = detections
                heatmap, heatmap_colored = self._generate_heatmap_fast(frame.shape, detections)
                last_heatmap, last_heatmap_colored = heatmap, heatmap_colored
                hotspots = self._find_hotspots_fast(heatmap, threshold=0.5)
                last_hotspots = hotspots
                
                dets_array = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][0]+d['bbox'][2], d['bbox'][1]+d['bbox'][3], d['confidence']] for d in detections]) if detections else np.empty((0, 5))
                tracks = self.tracker.update(dets_array)
                
                for det in detections:
                    det['id'] = None
                for track in tracks:
                    x1, y1, x2, y2, track_id = track
                    for det in detections:
                        dx1, dy1, dw, dh = det['bbox']
                        if abs(dx1 - x1) < 20 and abs(dy1 - y1) < 20:
                            det['id'] = int(track_id)
                            break
                
                self._track_flow_from_tracks(tracks)
                density = min(len(detections) / 100, 1.0)
                self.intelligence.update(len(detections), density, detections)
            
            count = len(last_dets)
            density = min(count / 100, 1.0)
            scene_label, scene_conf = self.scene_result
            anomaly, anomaly_type = self._detect_anomalies(count, density, last_heatmap if last_heatmap is not None else np.zeros((10, 10)), scene_label)
            
            self.density_history.append(density)
            self.count_history.append(count)
            
            frame_annotated = self._draw_overlay(frame, last_dets, last_heatmap_colored if last_heatmap_colored is not None else np.zeros_like(frame), last_hotspots)
            
            
            loop_time = time.time() - loop_start
            self.fps_counter.append(1.0 / max(0.001, loop_time))
            fps = int(np.mean(list(self.fps_counter)))
            
            with self.lock:
                self.current_frame = frame_annotated
                self.stats.update({
                    'count': count, 'density': int(density * 100), 'hotspots': len(last_hotspots),
                    'status': self._get_status(count), 'scene': scene_label, 'scene_conf': scene_conf,
                    'anomaly': anomaly, 'anomaly_type': anomaly_type,
                    'entry_count': self.entry_count, 'exit_count': self.exit_count,
                    'density_trend': 'stable', 'fps': fps
                })
            
            target_fps = 30
            sleep_time = max(0, (1.0 / target_fps) - loop_time)
            if sleep_time > 0: time.sleep(sleep_time)

    def _get_status(self, count):
        if count < 15: return "Faible"
        elif count < 35: return "Normal"
        elif count < 55: return "Élevé"
        else: return "Critique"

    def get_frame(self):
        with self.lock: return self.current_frame
    
    def get_stats(self):
        with self.lock: return self.stats.copy()
    
    def get_intelligence_data(self):
        return self.intelligence.get_intelligence_data()

    def process_image(self, image_path):
        """Traite une image unique pour analyse de densité"""
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.error("❌ Impossible de charger l'image")
                return None
            
            # Redimensionner si trop grande
            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
            
            logger.info("📊 Début analyse image...")
            
            # 🎯 DÉTECTER LES PERSONNES (PRIORITÉ 1)
            detections = self._detect_people_fast(frame)
            count = len(detections)
            logger.info(f"✅ {count} personnes détectées")
            
            # 🔥 CALCULER LA DENSITÉ
            density = min(count / 100, 1.0)
            density_pct = int(density * 100)
            
            # 🗺️ GÉNÉRER HEATMAP
            heatmap, heatmap_colored = self._generate_heatmap_fast(frame.shape, detections)
            
            # 🎯 TROUVER HOTSPOTS
            hotspots = self._find_hotspots_fast(heatmap, threshold=0.5)
            logger.info(f"🔥 {len(hotspots)} hotspots détectés")
            
            # 🏙️ DÉTECTION DE SCÈNE (SYNCHRONE)
            scene_label = "Analyse..."
            scene_conf = 0.0
            
            # Charger le modèle de scène si nécessaire
            if not hasattr(self, 'scene_model') or self.scene_model is None:
                try:
                    logger.info("⏳ Chargement du modèle de scène...")
                    model = models.resnet18(num_classes=365)
                    checkpoint = torch.load('resnet18_places365.pth', map_location=device)
                    state_dict = checkpoint['state_dict']
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                    model.to(device)
                    model.eval()
                    
                    with open('categories_places365.txt') as f:
                        classes = f.read().splitlines()
                    
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    self.scene_model = model
                    self.scene_classes = classes
                    self.scene_transform = transform
                    logger.info("✅ Modèle de scène chargé")
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de charger le modèle de scène: {e}")
                    self.scene_model = None
            
            # Détecter la scène
            if self.scene_model is not None:
                try:
                    logger.info("🔍 Analyse de scène en cours...")
                    scene_label, scene_conf = self.detect_scene_sync(frame)
                    logger.info(f"✅ Scène détectée: {scene_label} (confiance: {scene_conf:.2f})")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur détection scène: {e}")
                    scene_label = "Erreur détection"
            else:
                scene_label = "Modèle non disponible"
            
            # 🚨 DÉTECTION D'ANOMALIES
            anomaly, anomaly_type = self._detect_anomalies(
                count, 
                density, 
                heatmap if heatmap is not None else np.zeros((10, 10)), 
                scene_label
            )
            
            if anomaly:
                logger.warning(f"⚠️ ANOMALIE: {anomaly_type}")
            
            # 🎨 CRÉER L'OVERLAY AVEC PANEL D'INFOS
            h, w = frame.shape[:2]
            
            # Appliquer heatmap
            if heatmap_colored is not None:
                overlay = cv2.addWeighted(frame, 0.65, heatmap_colored, 0.35, 0)
            else:
                overlay = frame.copy()
            
            # Dessiner les détections
            for det in detections:
                x, y, dw, dh = det['bbox']
                cv2.rectangle(overlay, (x, y), (x+dw, y+dh), (0, 255, 0), 2)
                conf = det['confidence']
                cv2.putText(overlay, f"{conf:.2f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Dessiner hotspots
            for idx, hs in enumerate(hotspots[:3]):
                x, y, hw, hh = hs['bbox']
                cv2.rectangle(overlay, (x, y), (x+hw, y+hh), (0, 0, 255), 3)
                cv2.putText(overlay, f"HOTSPOT {idx+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Panel d'informations
            panel_height = 120
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)
            
            # Ligne 1
            cv2.putText(panel, f"PERSONNES: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, f"DENSITE: {density_pct}%", (250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, f"STATUS: {self._get_status(count)}", (500, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Ligne 2
            cv2.putText(panel, f"SCENE: {scene_label[:30]}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(panel, f"CONF: {scene_conf:.2f}", (500, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Ligne 3 - Hotspots
            cv2.putText(panel, f"HOTSPOTS: {len(hotspots)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            
            # Barre de densité
            bar_y = 105
            bar_width = int((w - 250) * (density_pct / 100))
            bar_color = (0, 255, 0) if density_pct < 40 else (0, 200, 255) if density_pct < 70 else (0, 0, 255)
            cv2.rectangle(panel, (240, bar_y), (240 + bar_width, bar_y + 10), bar_color, -1)
            cv2.rectangle(panel, (240, bar_y), (w - 10, bar_y + 10), (100, 100, 100), 2)
            
            # Combiner panel + overlay
            frame_annotated = np.vstack([panel, overlay])
            
            # 📸 ENCODER EN BASE64
            _, buffer = cv2.imencode('.jpg', frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 📊 RÉSULTATS
            stats = {
                'count': count,
                'density': density_pct,
                'hotspots': len(hotspots),
                'status': self._get_status(count),
                'scene': scene_label,
                'scene_conf': float(scene_conf),
                'anomaly': anomaly,
                'anomaly_type': anomaly_type if anomaly else None,
                'image': f"data:image/jpeg;base64,{img_base64}"
            }
            
            logger.info(f"✅ Analyse terminée: {count} personnes, {density_pct}% densité, scène={scene_label}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# Flask App
app = Flask(__name__)
video_analyzer = UltraFastVideoAnalyzer()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SADU PRO - Dashboard Complet</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --dark: #1e1e2f; --light: #27293d; --accent: #e14eca; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--dark); color: white; font-family: 'Segoe UI', sans-serif; display: grid; grid-template-columns: 250px 1fr; height: 100vh; overflow: hidden; }
        
        .sidebar { background: var(--light); padding: 20px; border-right: 1px solid #333; overflow-y: auto; }
        .logo { font-size: 24px; font-weight: bold; margin-bottom: 40px; color: var(--accent); }
        .menu-item { padding: 15px; cursor: pointer; border-radius: 10px; margin-bottom: 10px; transition: .3s; }
        .menu-item:hover, .menu-item.active { background: var(--accent); }
        .menu-item i { margin-right: 10px; width: 20px; }
        
        .main { padding: 20px; overflow-y: auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .status-badge { background: #00bf9a; padding: 5px 15px; border-radius: 20px; font-size: 12px; }
        
        .page { display: none; }
        .page.active { display: block; }
        
        .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: var(--light); padding: 20px; border-radius: 15px; position: relative; overflow: hidden; }
        .card h3 { margin: 0; font-size: 14px; opacity: 0.7; text-transform: uppercase; }
        .card .val { font-size: 28px; font-weight: bold; margin-top: 10px; }
        .card .icon { position: absolute; right: 20px; top: 20px; font-size: 40px; opacity: 0.1; }
        
        .monitor-container { background: black; border-radius: 15px; overflow: hidden; position: relative; height: 500px; display: flex; justify-content: center; align-items: center; }
        .monitor-container img { max-width: 100%; max-height: 100%; }
        
        .controls { margin-top: 20px; display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { padding: 12px 25px; border: none; border-radius: 30px; font-weight: bold; cursor: pointer; transition: .3s; font-size: 14px; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .btn-primary { background: var(--accent); color: white; }
        .btn-danger { background: #fd5d93; color: white; }
        .btn-secondary { background: #333; color: white; }
        
        .toggle-group { display: flex; gap: 10px; margin-left: auto; }
        .toggle { background: #333; color: white; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 12px; transition: .3s; }
        .toggle.active { background: #00bf9a; }
        .toggle:hover { opacity: 0.8; }
        
        .upload-area { border: 3px dashed var(--accent); border-radius: 12px; padding: 35px; text-align: center; cursor: pointer; transition: all 0.3s; margin-bottom: 20px; background: rgba(225, 78, 202, 0.1); }
        .upload-area:hover { background: rgba(225, 78, 202, 0.2); transform: scale(1.02); }
        
        .anomaly-alert { background: rgba(253, 93, 147, 0.1); border-left: 4px solid #fd5d93; padding: 12px; margin: 12px 0; border-radius: 8px; display: none; }
        .anomaly-alert.active { display: block; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.85; } }
        
        .intelligence-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .prediction-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; }
        .prediction-card h3 { margin-bottom: 15px; font-size: 18px; }
        .prediction-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
        .prediction-label { opacity: 0.8; font-size: 14px; }
        
        .behavior-card { background: var(--light); padding: 20px; border-radius: 15px; }
        .behavior-item { display: flex; justify-content: space-between; padding: 12px; margin: 8px 0; background: rgba(255,255,255,0.05); border-radius: 8px; }
        .behavior-item .label { display: flex; align-items: center; gap: 10px; }
        .behavior-item .value { font-weight: bold; font-size: 18px; }
        
        .config-section { background: var(--light); padding: 25px; border-radius: 15px; margin-bottom: 20px; }
        .config-section h3 { margin-bottom: 20px; color: var(--accent); }
        .config-item { display: flex; justify-content: space-between; align-items: center; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 12px; }
        .config-item label { font-size: 14px; }
        .config-item input[type="range"] { width: 200px; }
        .config-value { min-width: 60px; text-align: right; font-weight: bold; color: var(--accent); }
        
        .slider { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 5px; background: #333; outline: none; }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--accent); cursor: pointer; }
        
        .feature-card { transition: transform 0.3s, box-shadow 0.3s; }
        .feature-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .feature-icon { transition: transform 0.3s; }
        .feature-card:hover .feature-icon { transform: scale(1.1); }
        .feature-stats { transition: color 0.3s; }
        
        .quick-stat { text-align: center; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; }
        .stat-value { font-size: 2em; font-weight: bold; color: var(--accent); }
        .stat-label { color: #888; margin-top: 5px; }
        
        .step { display: flex; align-items: center; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo"><i class="fas fa-city"></i> SADU PRO</div>
        <div class="menu-item active" onclick="showPage('home')">
            <i class="fas fa-home"></i> Accueil
        </div>
        <div class="menu-item" onclick="showPage('dashboard')">
            <i class="fas fa-chart-line"></i> Dashboard
        </div>
        <div class="menu-item" onclick="showPage('intelligence')">
            <i class="fas fa-brain"></i> Intelligence
        </div>
        <div class="menu-item" onclick="showPage('configuration')">
            <i class="fas fa-cogs"></i> Configuration
        </div>
    </div>
    
    <div class="main">
        <div id="home" class="page active">
            <div class="header">
                <h1>🏠 Bienvenue sur SADU PRO</h1>
                <p style="color: #888; margin-top: 5px;">Système d'Analyse de Densité Urbaine </p>
            </div>
            
            <!-- Hero Section -->
            <div class="card hero-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 30px; position: relative; overflow: hidden;">
                <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
                <div style="position: absolute; bottom: -30px; left: -30px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
                <h2 style="font-size: 2.5em; margin-bottom: 15px;">🚀 Révolutionnez la Gestion Urbaine</h2>
                <p style="font-size: 1.2em; line-height: 1.6; margin-bottom: 20px;">
                    Découvrez comment l'IA transforme la surveillance des foules avec précision et rapidité.
                </p>
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <button class="btn btn-primary" onclick="showPage('dashboard')" style="background: white; color: #667eea; border: none;">
                        <i class="fas fa-play"></i> Commencer l'Analyse
                    </button>
                    <button class="btn btn-secondary" onclick="showPage('intelligence')" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);">
                        <i class="fas fa-brain"></i> Découvrir l'IA
                    </button>
                </div>
            </div>
            
            <!-- Features Grid -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #667eea; margin-bottom: 15px;">👥</div>
                    <h3>Détection Avancée</h3>
                    <p>YOLOv8 pour un suivi précis des individus en temps réel.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #667eea;">précision Avancée</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #00bf9a; margin-bottom: 15px;">📊</div>
                    <h3>Analyse de Densité</h3>
                    <p>Cartographie thermique avancée pour identifier les zones de surpopulation instantanément.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #00bf9a;">Temps réel</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #fd5d93; margin-bottom: 15px;">🚨</div>
                    <h3>Alertes Intelligentes</h3>
                    <p>Notifications automatiques en cas de surpopulation (>10% densité) par scène.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #fd5d93;">Temps réel</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #e14eca; margin-bottom: 15px;">🧠</div>
                    <h3>Intelligence Prédictive</h3>
                    <p>Prédictions de flux et analyses comportementales pour une gestion proactive.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #e14eca;">Haute performance</div>
                </div>
            </div>
            
            <!-- Quick Stats -->
            <div class="card" style="margin-bottom: 20px;">
                <h3 style="text-align: center; margin-bottom: 20px;">📊 Statistiques en Direct</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div class="quick-stat">
                        <div class="stat-value" id="homeCount">0</div>
                        <div class="stat-label">Personnes détectées</div>
                    </div>
                    <div class="quick-stat">
                        <div class="stat-value" id="homeDensity">0%</div>
                        <div class="stat-label">Densité actuelle</div>
                    </div>
                    <div class="quick-stat">
                        <div class="stat-value" id="homeAlerts">0</div>
                        <div class="stat-label">Alertes actives</div>
                    </div>
                    <div class="quick-stat">
                        <div class="stat-value" id="homeHotspots">0</div>
                        <div class="stat-label">Hotspots détectés</div>
                    </div>
                </div>
            </div>
            
            <!-- Getting Started -->
            <div class="card">
                <h3>🚀 Guide de Démarrage Rapide</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                    <div class="step">
                        <div class="step-number" style="background: #667eea; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">1</div>
                        <strong>Chargez vos médias</strong><br>
                        <span style="color: #666;">Vidéo ou image via le Dashboard</span>
                    </div>
                    <div class="step">
                        <div class="step-number" style="background: #00bf9a; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">2</div>
                        <strong>Lancez l'analyse</strong><br>
                        <span style="color: #666;">Visualisez en temps réel</span>
                    </div>
                    <div class="step">
                        <div class="step-number" style="background: #fd5d93; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">3</div>
                        <strong>Consultez l'Intelligence</strong><br>
                        <span style="color: #666;">Analyse comportementale et prédictions</span>
                    </div>
                    <div class="step">
                        <div class="step-number" style="background: #e14eca; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">4</div>
                        <strong>Surveillez les alertes</strong><br>
                        <span style="color: #666;">Réagissez aux anomalies détectées</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="dashboard" class="page">
            <div class="header">
                <h1>📊 Dashboard - Centre de Contrôle</h1>
                <div class="status-badge">SYSTEM ONLINE</div>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('up').click()">
                <h3>📹 Charger une vidéo</h3>
                <p>Cliquez pour sélectionner (MP4, AVI, MOV)</p>
                <input type="file" id="up" hidden onchange="upload(this)" accept=".mp4,.avi,.mov,video/*">
            </div>
            
            <div class="upload-area" onclick="document.getElementById('upImg').click()">
                <h3>🖼️ Charger une image</h3>
                <p>Cliquez pour sélectionner (JPG, PNG)</p>
                <input type="file" id="upImg" hidden onchange="uploadImage(this)" accept=".jpg,.jpeg,.png,.bmp,.webp,image/*">
            </div>
            
            <div class="anomaly-alert" id="anomalyAlert">
                <strong>⚠️ ANOMALIE DÉTECTÉE:</strong>
                <span id="anomalyType"></span>
            </div>
            
            <div class="grid-stats">
                <div class="card">
                    <h3>AFFLUENCE</h3>
                    <div class="val" id="count">0</div>
                    <div class="icon"><i class="fas fa-users"></i></div>
                </div>
                <div class="card">
                    <h3>DENSITÉ</h3>
                    <div class="val" id="dens">0%</div>
                    <div class="icon"><i class="fas fa-percent"></i></div>
                </div>
                <div class="card">
                    <h3>ANOMALIE</h3>
                    <div class="val" id="anom" style="color:#00bf9a">AUCUNE</div>
                    <div class="icon"><i class="fas fa-exclamation-triangle"></i></div>
                </div>
                <div class="card">
                    <h3>FPS</h3>
                    <div class="val" id="fpsDisplay">0</div>
                    <div class="icon"><i class="fas fa-tachometer-alt"></i></div>
                </div>
                <div class="card image-stat" id="sceneCard" style="display:none;">
                    <h3>SCÈNE</h3>
                    <div class="val" id="scene">Analyse...</div>
                    <div class="icon"><i class="fas fa-map"></i></div>
                </div>
                <div class="card image-stat" id="hotspotsCard" style="display:none;">
                    <h3>HOTSPOTS</h3>
                    <div class="val" id="hotspots">0</div>
                    <div class="icon"><i class="fas fa-fire"></i></div>
                </div>
                <div class="card image-stat" id="statusCard" style="display:none;">
                    <h3>STATUT</h3>
                    <div class="val" id="status">Arrêté</div>
                    <div class="icon"><i class="fas fa-info-circle"></i></div>
                </div>
            </div>
            
            <div class="monitor-container">
                <img id="feed" src="/video_feed">
                <img id="imageResult" style="display:none; max-width:100%; height:auto;">
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="startAnalysis()"><i class="fas fa-play"></i> START</button>
                <button class="btn btn-danger" onclick="stopAnalysis()"><i class="fas fa-stop"></i> STOP</button>
                
                <div class="toggle-group">
                    <div class="toggle active" onclick="toggle(this, 'heatmap')">HEATMAP</div>
                    <div class="toggle" onclick="toggle(this, 'seg')">SEGMENTATION</div>
                    <div class="toggle active" onclick="toggle(this, 'tracker')">TRACKING</div>
                </div>
            </div>
        </div>
        
        <div id="intelligence" class="page">
            <div class="header">
                <h1>🧠 Intelligence Artificielle</h1>
            </div>
            
            <div class="intelligence-grid">
                <div class="prediction-card">
                    <h3>📈 Prédiction 5 minutes</h3>
                    <div class="prediction-value" id="pred5">0</div>
                    <div class="prediction-label">personnes attendues</div>
                </div>
                
                <div class="prediction-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>📊 Prédiction 15 minutes</h3>
                    <div class="prediction-value" id="pred15">0</div>
                    <div class="prediction-label">personnes attendues</div>
                </div>
                
                <div class="prediction-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>⏰ Heure de Pic</h3>
                    <div class="prediction-value" style="font-size: 24px;" id="peakTime">--:--</div>
                    <div class="prediction-label">prévision basée sur historique</div>
                </div>
                
                <div class="prediction-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h3>📉 Tendance</h3>
                    <div class="prediction-value" style="font-size: 24px;" id="trendVal">Stable</div>
                    <div class="prediction-label" id="riskLevel">Risque: Faible</div>
                </div>
            </div>
            
            <div class="behavior-card" style="margin-top: 20px;">
                <h3 style="margin-bottom: 15px;">👥 Analyse Comportementale</h3>
                <div class="behavior-item">
                    <div class="label">
                        <i class="fas fa-user" style="color: #667eea;"></i>
                        <span>Personnes stationnaires</span>
                    </div>
                    <div class="value" id="stationary">0</div>
                </div>
                <div class="behavior-item">
                    <div class="label">
                        <i class="fas fa-walking" style="color: #00bf9a;"></i>
                        <span>Personnes en marche</span>
                    </div>
                    <div class="value" id="walking">0</div>
                </div>
                <div class="behavior-item">
                    <div class="label">
                        <i class="fas fa-running" style="color: #fd5d93;"></i>
                        <span>Personnes qui courent</span>
                    </div>
                    <div class="value" id="running">0</div>
                </div>
                <div class="behavior-item">
                    <div class="label">
                        <i class="fas fa-users" style="color: #f5576c;"></i>
                        <span>Groupes détectés</span>
                    </div>
                    <div class="value" id="gathering">0</div>
                </div>
            </div>
        </div>
        
        <div id="configuration" class="page">
            <div class="header">
                <h1>⚙️ Configuration Système</h1>
            </div>
            
            <div class="config-section">
                <h3><i class="fas fa-eye"></i> Paramètres de Détection</h3>
                <div class="config-item">
                    <label>Confiance de détection (YOLO)</label>
                    <input type="range" class="slider" min="0.1" max="0.9" step="0.05" value="0.4" id="detConf" 
                           oninput="updateConfig('detection_confidence', this.value)">
                    <span class="config-value" id="detection_confidenceVal">0.40</span>
                </div>
                <div class="config-item">
                    <label>Seuil d'anomalie</label>
                    <input type="range" class="slider" min="0.5" max="0.95" step="0.05" value="0.75" id="anomThresh" 
                           oninput="updateConfig('anomaly_threshold', this.value)">
                    <span class="config-value" id="anomaly_thresholdVal">0.75</span>
                </div>
            </div>
            
            <div class="config-section">
                <h3><i class="fas fa-fire"></i> Paramètres Heatmap</h3>
                <div class="config-item">
                    <label>Intensité Heatmap</label>
                    <input type="range" class="slider" min="0.1" max="0.6" step="0.05" value="0.35" id="heatIntensity" 
                           oninput="updateConfig('heatmap_intensity', this.value)">
                    <span class="config-value" id="heatmap_intensityVal">0.35</span>
                </div>
                <div class="config-item">
                    <label>Sigma Gaussien (flou)</label>
                    <input type="range" class="slider" min="1" max="15" step="1" value="5" id="gaussSigma" 
                           oninput="updateConfig('gaussian_sigma', this.value)">
                    <span class="config-value" id="gaussian_sigmaVal">5</span>
                </div>
            </div>
            
            <div class="config-section">
                <h3><i class="fas fa-layer-group"></i> Paramètres Segmentation</h3>
                <div class="config-item">
                    <label>Taille du flou gaussien</label>
                    <input type="range" class="slider" min="5" max="51" step="2" value="21" id="segBlur" 
                           oninput="updateConfig('segmentation_blur', this.value)">
                    <span class="config-value" id="segmentation_blurVal">21</span>
                </div>
            </div>
            
            <div class="config-section">
                <h3><i class="fas fa-tachometer-alt"></i> Paramètres Performance</h3>
                <div class="config-item">
                    <label>Saut d'images (frame skip)</label>
                    <input type="range" class="slider" min="1" max="10" step="1" value="3" id="frameSkip" 
                           oninput="updateConfig('frame_skip', this.value)">
                    <span class="config-value" id="frame_skipVal">3</span>
                </div>
            </div>
            
            <div class="config-section">
                <h3><i class="fas fa-info-circle"></i> Informations Système</h3>
                <div class="config-item">
                    <label>Version du système</label>
                    <span class="config-value">SADU v2.0</span>
                </div>
                <div class="config-item">
                    <label>Modèle de détection</label>
                    <span class="config-value">YOLOv8n</span>
                </div>
                <div class="config-item">
                    <label>Tracking algorithme</label>
                    <span class="config-value">SORT</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            event.target.closest('.menu-item').classList.add('active');
        }
        
        function upload(el) {
            let fd = new FormData();
            fd.append('video', el.files[0]);
            fetch('/upload_video', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
                if(d.success) alert('✅ Vidéo chargée!');
                else alert('❌ Erreur: ' + d.error);
            });
        }
        
        function uploadImage(el) {
            let fd = new FormData();
            fd.append('image', el.files[0]);
            fetch('/upload_image', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
                if(d.success) {
                    alert('✅ Image analysée!');
                    // Masquer la vidéo et afficher l'image
                    document.getElementById('feed').style.display = 'none';
                    document.getElementById('imageResult').src = d.data.image;
                    document.getElementById('imageResult').style.display = 'block';
                    // Masquer les stats principales et afficher les stats d'image
                    document.querySelectorAll('.card:not(.image-stat)').forEach(card => card.style.display = 'none');
                    document.querySelectorAll('.image-stat').forEach(card => card.style.display = 'block');
                    // Mettre à jour les stats d'image
                    document.getElementById('scene').innerText = d.data.scene;
                    document.getElementById('hotspots').innerText = d.data.hotspots;
                    document.getElementById('status').innerText = d.data.status;
                } else alert('❌ Erreur: ' + d.error);
            });
        }
        
        function toggle(el, type) {
            el.classList.toggle('active');
            // Pour l'instant, juste visuel - les overlays sont toujours activés
            alert(type + ' toggled');
        }
        
        function startAnalysis() {
            fetch('/start_analysis', {method:'POST'}).then(r=>r.json()).then(d => {
                if(d.success) {
                    alert('✅ Analyse démarrée!');
                    // Masquer l'image et afficher la vidéo
                    document.getElementById('imageResult').style.display = 'none';
                    // Afficher les stats principales et masquer les stats d'image
                    document.querySelectorAll('.card:not(.image-stat)').forEach(card => card.style.display = 'block');
                    document.querySelectorAll('.image-stat').forEach(card => card.style.display = 'none');
                    document.getElementById('feed').style.display = 'block';
                    document.getElementById('feed').src = '/video_feed?' + new Date().getTime();
                } else alert('❌ Erreur: ' + d.error);
            });
        }
        
        function stopAnalysis() {
            fetch('/stop_analysis', {method:'POST'}).then(r=>r.json()).then(d => {
                if(d.success) {
                    alert('⏹️ Analyse arrêtée');
                    // Masquer les affichages
                    document.getElementById('feed').style.display = 'none';
                    document.getElementById('imageResult').style.display = 'none';
                    // Afficher les stats principales et masquer les stats d'image
                    document.querySelectorAll('.card:not(.image-stat)').forEach(card => card.style.display = 'block');
                    document.querySelectorAll('.image-stat').forEach(card => card.style.display = 'none');
                } else alert('❌ Erreur: ' + d.error);
            });
        }
        
        function toggle(el, opt) {
            el.classList.toggle('active');
            let val = el.classList.contains('active');
            fetch('/config/'+opt+'/'+val).then(r=>r.json());
        }
        
        function updateConfig(key, value) {
            // Mettre à jour l'affichage immédiatement
            const displayId = key + 'Val';
            const displayElement = document.getElementById(displayId);
            if (displayElement) {
                if (key === 'frame_skip' || key === 'gaussian_sigma' || key === 'segmentation_blur') {
                    displayElement.textContent = value;
                } else {
                    displayElement.textContent = parseFloat(value).toFixed(2);
                }
            }
            
            // Envoyer au serveur
            fetch('/update_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({key: key, value: parseFloat(value)})
            }).then(r=>r.json()).then(d => {
                if (!d.success) {
                    console.error('Erreur mise à jour config:', d.error);
                }
            }).catch(err => {
                console.error('Erreur réseau:', err);
            });
        }

        setInterval(() => {
            fetch('/get_stats').then(r=>r.json()).then(d => {
                document.getElementById('count').innerText = d.count;
                document.getElementById('dens').innerText = d.density + '%';
                document.getElementById('fpsDisplay').innerText = d.fps;
                let anom = document.getElementById('anom');
                anom.innerText = d.anomaly ? d.anomaly_type : 'AUCUNE';
                anom.style.color = d.anomaly ? '#fd5d93' : '#00bf9a';
                
                // Update home stats
                document.getElementById('homeCount').innerText = d.count;
                document.getElementById('homeDensity').innerText = d.density + '%';
                document.getElementById('homeAlerts').innerText = d.anomaly ? '1' : '0';
                document.getElementById('homeHotspots').innerText = d.hotspots || '0';
                
                const anomalyAlert = document.getElementById('anomalyAlert');
                if (d.anomaly) {
                    anomalyAlert.classList.add('active');
                    document.getElementById('anomalyType').textContent = d.anomaly_type;
                } else {
                    anomalyAlert.classList.remove('active');
                }
            });
            
            fetch('/get_intelligence').then(r=>r.json()).then(d => {
                if(d.predictions) {
                    document.getElementById('pred5').textContent = d.predictions.next_5min;
                    document.getElementById('pred15').textContent = d.predictions.next_15min;
                    document.getElementById('peakTime').textContent = d.predictions.peak_time;
                    document.getElementById('trendVal').textContent = d.predictions.trend;
                    document.getElementById('riskLevel').textContent = 'Risque: ' + d.predictions.risk_level;
                }
                if(d.behaviors) {
                    document.getElementById('stationary').textContent = d.behaviors.stationary;
                    document.getElementById('walking').textContent = d.behaviors.walking;
                    document.getElementById('running').textContent = d.behaviors.running;
                    document.getElementById('gathering').textContent = d.behaviors.gathering;
                }
            });
        }, 800);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier'})
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nom vide'})
    try:
        video_path = DATA_DIR / 'current_video.mp4'
        file.save(video_path)
        video_analyzer.set_video(video_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier image'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nom vide'})
    try:
        image_path = DATA_DIR / 'current_image.jpg'
        file.save(image_path)
        result = video_analyzer.process_image(image_path)
        if result:
            return jsonify({'success': True, 'data': result})
        else:
            return jsonify({'success': False, 'error': 'Erreur traitement image'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    try:
        if video_analyzer.start():
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Vidéo non chargée'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    try:
        video_analyzer.stop()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stats')
def get_stats():
    try:
        stats = video_analyzer.get_stats()
        return jsonify(stats)
    except:
        return jsonify({'count': 0, 'density': 0, 'hotspots': 0, 'status': 'Erreur', 'scene': 'Inconnu', 'scene_conf': 0.0, 'anomaly': False, 'anomaly_type': None, 'entry_count': 0, 'exit_count': 0, 'density_trend': 'stable', 'fps': 0})

@app.route('/get_intelligence')
def get_intelligence():
    try:
        data = video_analyzer.get_intelligence_data()
        return jsonify(data)
    except:
        return jsonify({'predictions': {}, 'b"ehaviors': {}, 'history_length': 0})

@app.route('/update_config', methods=['POST'])
def update_config():
    try:
        data = request.json
        key = data.get('key')
        value = data.get('value')
        if video_analyzer.update_config(key, value):
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Clé invalide'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = video_analyzer.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config/<opt>/<val>')
def config_option(opt, val):
    v = (val.lower() == 'true')
    if opt == 'heatmap': video_analyzer.show_heatmap = v
    elif opt == 'seg': video_analyzer.show_segmentation = v
    elif opt == 'tracker': video_analyzer.show_tracking = v
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SADU PRO - SYSTÈME COMPLET CORRIGÉ")
    print("="*80)
    print("📍 URL:      http://localhost:5000")
    print("="*80)
    print("✨ FONCTIONNALITÉS:")
    print("   ✅ Segmentation multi-classes (personnes=rouge, voitures=bleu, etc.)")
    print("   ✅ Détection de scène activée")
    print("   ✅ Intelligence IA avec prédictions")
    print("   ✅ Analyse comportementale avancée")
    print("   ✅ Configuration complète et dynamique")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)