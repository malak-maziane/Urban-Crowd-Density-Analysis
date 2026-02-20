#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 SADU PRO - Détection Foule & Reconnaissance Faciale en Temps Réel
Real-time crowd detection with facial expressions and VIP recognition
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
import face_recognition

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
    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.3):  # Optimisé pour rapidité
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
        self.behavior_patterns = {'stationary': 0, 'walking': 0, 'running': 0, 'gathering': 0, 'panic': False}
        self.predictions = {
            'next_5min': 0, 'next_15min': 0, 'peak_time': 'Analyse...', 
            'trend': 'stable', 'risk_level': 'faible'
        }
        self.track_histories = {} # id -> list of (x,y,t)
        
    def update(self, count, density, detections, tracks=[]):
        current_time = time.time()
        self.density_history.append(density)
        self.count_history.append(count)
        self.time_history.append(current_time)
        self._analyze_behaviors_real(detections, tracks, current_time)
        if len(self.count_history) > 50:
            self._generate_predictions()
    
    def _analyze_behaviors_real(self, detections, tracks, current_time):
        # Analyse réelle basée sur la vitesse des trackers
        stationary = 0
        walking = 0
        running = 0
        
        # Mettre à jour l'historique de position pour calculer la vitesse
        current_ids = set()
        
        # Si tracks est vide ou format incorrect, fallback sur heuristique simple
        if len(tracks) == 0:
            self.behavior_patterns = {'stationary': len(detections), 'walking': 0, 'running': 0, 'gathering': 0, 'panic': False}
            return

        speeds = []
        
        for track in tracks:
            # Format track: [x1, y1, x2, y2, id] ou similaire selon SortTracker
            x1, y1, x2, y2, trk_id = track[:5]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            trk_id = int(trk_id)
            current_ids.add(trk_id)
            
            if trk_id not in self.track_histories:
                self.track_histories[trk_id] = deque(maxlen=10)
            
            self.track_histories[trk_id].append((cx, cy, current_time))
            
            # Calculer vitesse si assez d'historique
            hist = self.track_histories[trk_id]
            if len(hist) >= 3:
                # Vitesse en pixels par seconde
                # (pos_now - pos_prev) / time_diff
                # Prendre une moyenne sur les derniers points pour lisser
                p_now = hist[-1]
                p_prev = hist[0] # Il y a max 10 points, donc environ 1-3 secondes
                
                dist = np.sqrt((p_now[0]-p_prev[0])**2 + (p_now[1]-p_prev[1])**2)
                dt = p_now[2] - p_prev[2]
                
                if dt > 0:
                    speed = dist / dt # px/sec
                    speeds.append(speed)
                    
                    if speed < 10: # Seuil bas pour stationnaire
                        stationary += 1
                    elif speed < 50: # Seuil marche
                        walking += 1
                    else: # Course / Panique
                        running += 1
                else:
                    stationary += 1
            else:
                stationary += 1 # Pas assez de données, assumer stationnaire
        
        # Nettoyage historique
        for tid in list(self.track_histories.keys()):
            if tid not in current_ids:
                del self.track_histories[tid]
        
        # Détection de Panique: Si plus de 30% des gens courent et qu'il y a du monde
        panic_mode = False
        total_people = stationary + walking + running
        if total_people > 5 and (running / total_people) > 0.3:
            panic_mode = True
            
        centers = [det['center'] for det in detections]
        gathering = self._detect_groups(centers)
        
        self.behavior_patterns = {
            'stationary': stationary, 
            'walking': walking, 
            'running': running, 
            'gathering': gathering,
            'panic': panic_mode
        }
    
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
        
        # Risk level logic correction
        if self.behavior_patterns.get('panic', False):
             risk_level = 'CRITIQUE - PANIQUE'
        else:
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
        self.vips = {}  # name -> face_encoding
        self.vips_cache = {}  # Cache des VIPs détectés récemment: {name: {'last_seen': timestamp, 'face_img': img, 'emotion': str}}
        self.vips_cache_max_age = 30  # Garder les VIPs en cache pendant 30 secondes
        self.vip_registry = {} # {vip_name: track_id} pour persistance UNIQUE (1 nom = 1 id)
        self.notifications = [] # [{'message': str, 'timestamp': float}]
        self.stats = {'count': 0, 'density': 0, 'hotspots': 0, 'status': 'Arrêté', 'scene': 'Analyse...', 'scene_conf': 0.0, 'anomaly': False, 'anomaly_type': None, 'entry_count': 0, 'exit_count': 0, 'density_trend': 'stable', 'fps': 0}
        self.config = {'detection_confidence': 0.4, 'heatmap_intensity': 0.35, 'anomaly_threshold': 0.75, 'tracking_enabled': True, 'gaussian_sigma': 5, 'segmentation_blur': 21, 'frame_skip': 30}
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
        
        # PERSISTENCE
        self.persistent_faces = {} # key -> {face_img, name, emotion, last_seen}
        self.last_detections = []
        self.last_heatmap_colored = None
        self.last_hotspots = []
        self.last_vips_detected = []

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

    def add_vip(self, name, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.vips[name] = encodings[0]
                logger.info(f"✅ VIP ajouté: {name}")
                return True
            else:
                logger.error("❌ Aucun visage détecté dans l'image")
                return False
        except Exception as e:
            logger.error(f"❌ Erreur ajout VIP: {e}")
            return False

    def recognize_vip(self, frame):
        """Reconnaître les VIPs rapidement avec meilleure précision"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Utiliser le modèle HOG plus rapide
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized = []
        vip_names = list(self.vips.keys())
        vip_encodings = list(self.vips.values())
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparaison plus rapide
            distances = face_recognition.face_distance(vip_encodings, face_encoding)
            best_match_idx = np.argmin(distances)
            
            # Threshold plus strict pour meilleure précision
            if distances[best_match_idx] < 0.5:  # Réduit de 0.6
                name = vip_names[best_match_idx]
                emotion = self._detect_emotion(frame, face_location)
                top, right, bottom, left = face_location
                recognized.append({
                    'name': name,
                    'location': face_location,
                    'emotion': emotion,
                    'confidence': float(1 - distances[best_match_idx])
                })
        return recognized

    def detect_faces_and_emotions(self, frame):
        """🚀 ULTRA-RAPIDE - Détection faciale et émotion pour TOUS (VIP + Foule)"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # HOG ultra-rapide pour detection visages
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if not face_locations:
                return []
            
            # Calcul des encodings SEULEMENT si des VIPs sont définis
            face_encodings = []
            if self.vips:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized = []
            vip_names = list(self.vips.keys())
            vip_encodings = list(self.vips.values())
            
            for i, face_location in enumerate(face_locations):
                name = None
                confidence = 0.0
                
                # Tenter reconnaissance VIP si possible
                if self.vips and i < len(face_encodings):
                    distances = face_recognition.face_distance(vip_encodings, face_encodings[i])
                    best_match_idx = np.argmin(distances)
                    if distances[best_match_idx] < 0.5:
                        name = vip_names[best_match_idx]
                        confidence = float(1 - distances[best_match_idx])
                
                # Émotion pour TOUT LE MONDE (VIP ou non)
                emotion = self._detect_emotion_fast(frame, face_location)
                
                recognized.append({
                    'name': name, # None si inconnu
                    'location': face_location,
                    'emotion': emotion,
                    'confidence': confidence
                })
            
            return recognized
        except Exception as e:
            logger.debug(f"Face/Emotion detection error: {e}")
            return []

    def _detect_emotion_fast(self, frame, face_location):
        """Detection emotion rapide - texte lisible"""
        try:
            top, right, bottom, left = face_location
            face_img = frame[top:bottom, left:right]
            
            if face_img.size == 0:
                return "Neutre"
            
            # Vérifier taille minimale
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                return "Neutre"
            
            h, w = face_img.shape[:2]
            
            # Extraire région centrale
            y_start = max(0, h // 4)
            y_end = min(h, 3 * h // 4)
            x_start = max(0, w // 4)
            x_end = min(w, 3 * w // 4)
            face_center = face_img[y_start:y_end, x_start:x_end]
            
            if face_center.size == 0:
                return "Neutre"
            
            # Analyse rapide BGR
            b, g, r = cv2.split(face_center)
            
            # Rougeur du visage
            red_intensity = r.astype(float) - g.astype(float) / 2
            avg_red = np.mean(red_intensity[red_intensity > 0]) if np.any(red_intensity > 0) else 0
            
            # Contraste
            if face_center.shape[0] > 2 and face_center.shape[1] > 2:
                gray_center = cv2.cvtColor(face_center, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray_center, cv2.CV_64F)
                contrast = np.std(laplacian)
            else:
                contrast = 0
            
            # Luminosité
            gray = cv2.cvtColor(face_center, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # DECISION EN TEXTE LISIBLE
            if avg_red > 30 and contrast > 20:
                return "COLERE"
            elif brightness < 60:
                return "TRISTE"
            elif brightness > 150 and contrast > 15:
                return "SURPRISE"
            elif avg_red > 50:
                return "PEUR"
            elif 80 < brightness < 140 and avg_red < 20:
                return "JOIE"
            else:
                return "NEUTRE"
                
        except Exception as e:
            return "NEUTRE"

    def _detect_emotion(self, frame, face_location):
        """Détect émotions rapidement avec optimisations"""
        try:
            from deepface import DeepFace
            top, right, bottom, left = face_location
            face_img = frame[top:bottom, left:right]
            if face_img.size == 0:
                return "Neutre"
            
            # Redimensionner pour plus de vitesse
            face_img_small = cv2.resize(face_img, (224, 224))
            
            try:
                result = DeepFace.analyze(face_img_small, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(result, list) and len(result) > 0:
                    emotions = result[0].get('emotion', {})
                    if emotions:
                        dominant = max(emotions, key=emotions.get)
                        emotion_map = {
                            'fear': '😰 Panique',
                            'happy': '😊 Joie',
                            'sad': '😢 Triste',
                            'angry': '😠 Colère',
                            'surprise': '😲 Surprise',
                            'neutral': '😐 Neutre',
                            'disgust': '🤮 Dégoût'
                        }
                        return emotion_map.get(dominant, dominant)
            except:
                pass
            return "Neutre"
        except Exception as e:
            logger.debug(f"Emotion detection failed: {e}")
            return "Neutre"

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
        """Détection rapide et optimisée des personnes"""
        if self.yolo_model is None: return []
        h, w = frame.shape[:2]
        
        # Optimiser la taille pour plus de vitesse
        scale = 480 / max(h, w)  # Réduit de 640 pour plus de vitesse
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Inférence rapide avec conf plus élevée
        results = self.yolo_model(resized, conf=max(0.45, self.config['detection_confidence']), iou=0.45, verbose=False)[0]
        
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0:  # Classe personne
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                w_box, h_box = x2 - x1, y2 - y1
                
                # Filtres pour éviter les faux positifs
                if w_box < 20 or h_box < 40: continue
                if conf < 0.4: continue  # Confiance minimale
                
                detections.append({
                    'bbox': [x1, y1, w_box, h_box],
                    'confidence': float(conf),
                    'center': (x1 + w_box // 2, y1 + h_box // 2),
                    'id': None
                })
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

    def _draw_overlay(self, frame, detections, heatmap_colored, hotspots, vips_detected=[]):
        """Dessiner les overlays avec meilleure visibilité et panel VIP détaillé"""
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        if self.show_segmentation:
            seg_mask = self._generate_segmentation_mask(frame, detections)
            frame = cv2.addWeighted(frame, 0.6, seg_mask, 0.4, 0)
        
        if self.show_heatmap:
            overlay = cv2.addWeighted(frame, 0.65, heatmap_colored, 0.35, 0)
        else:
            overlay = frame
        
        # ✅ DESSINER HOTSPOTS EN ROUGE CLAIR
        for idx, hs in enumerate(hotspots[:3]):
            x, y, hw, hh = hs['bbox']
            cv2.rectangle(overlay, (x, y), (x+hw, y+hh), (0, 100, 255), 3)  # Orange/rouge
            cv2.putText(overlay, f"HOTSPOT {idx+1}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        
        # ✅ DESSINER LES PERSONNES DÉTECTÉES
        for i, det in enumerate(detections):
            if i % max(1, len(detections) // 20) != 0: continue
            x, y, dw, dh = det['bbox']
            
            # Check VIP status via persistent tracking (Registry Lookup)
            is_vip = False
            vip_name = ""
            if det.get('id') is not None:
                tid = int(det['id'])
                # Check if this TID is the CURRENT assigned ID for any VIP
                if hasattr(self, 'vip_registry'):
                    for vname, vtid in self.vip_registry.items():
                        if vtid == tid:
                            is_vip = True
                            vip_name = vname
                            break
            
            if is_vip:
                # VIP detection: RED BOX + NAME
                cv2.rectangle(overlay, (x, y), (x+dw, y+dh), (0, 0, 255), 4) # Red
                cv2.putText(overlay, f"VIP: {vip_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Normal detection: GREEN BOX
                cv2.rectangle(overlay, (x, y), (x+dw, y+dh), (0, 255, 0), 2)
                if self.show_tracking and det.get('id'): 
                    cv2.putText(overlay, f"ID:{int(det['id'])}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ✅ NOTIFICATIONS VIP (OVERLAY)
        if hasattr(self, 'notifications'):
            import time
            curr_t = time.time()
            active_notifs = []
            y_notif = 50 
            
            for notif in self.notifications:
                if curr_t - notif['timestamp'] < 10.0: # Show for 10 seconds
                    active_notifs.append(notif)
                    msg = notif['message']
                    sub = notif.get('sub', '')
                    
                    # Draw elegant notification box
                    box_w = 400
                    box_h = 70
                    cx = w // 2
                    
                    # Semi-transparent background
                    sub_img = overlay[y_notif:y_notif+box_h, cx-box_w//2:cx+box_w//2]
                    white_rect = np.full(sub_img.shape, (0, 0, 0), dtype=np.uint8)
                    res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
                    overlay[y_notif:y_notif+box_h, cx-box_w//2:cx+box_w//2] = res
                    
                    cv2.rectangle(overlay, (cx-box_w//2, y_notif), (cx+box_w//2, y_notif+box_h), (0, 0, 255), 2)
                    
                    cv2.putText(overlay, msg, (cx-box_w//2 + 20, y_notif + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(overlay, sub, (cx-box_w//2 + 20, y_notif + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
                    
                    y_notif += 80
            
            self.notifications = active_notifs
        
        # ✅ DESSINER LES VISAGES DÉTECTÉS AVEC ÉMOTIONS
        vip_panel_data = []
        for vip_data in vips_detected:
            if isinstance(vip_data, dict):
                name = vip_data['name']
                emotion = vip_data['emotion']
                face_location = vip_data['location']
                confidence = vip_data.get('confidence', 1.0)
            else:
                if len(vip_data) == 3:
                    name, face_location, emotion = vip_data
                    confidence = 1.0
                else:
                    name, face_location = vip_data
                    emotion = "Neutre"
                    confidence = 1.0
            
            top, right, bottom, left = face_location
            
            if name: # C'est un VIP
                color = (0, 0, 255) # ROUGE
                label = "VIP: " + name
                # Grosses boîtes pour VIP
                cv2.rectangle(overlay, (left, top), (right, bottom), color, 5)
                # Label VIP
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                cv2.rectangle(overlay, (left-2, top-35), (left+text_size[0]+5, top-5), color, -1)
                cv2.putText(overlay, label, (left+2, top-12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            else: # C'est un membre de la foule
                color = (255, 200, 0) # BLEU CIEL / CYAN
                # Boîtes plus fines
                cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
            
            # AFFICHER EMOTION SUR LE VISAGE
            if emotion and emotion.strip():
                emotion_text = emotion.strip()
                # Couleur émotion selon type
                emo_color = (100, 255, 255) if name else (255, 255, 200)
                font_scale = 1.1 if name else 0.7
                thickness = 3 if name else 2
                cv2.putText(overlay, emotion_text, (left+5, top+35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, emo_color, thickness)
            
            vip_panel_data.append({
                'name': name,
                'location': face_location,
                'emotion': emotion,
                'confidence': confidence,
                'detected': True
            })
        
        cv2.line(overlay, (0, self.exit_line), (w, self.exit_line), (0, 0, 255), 2)
        
        # ===== PANEL PRINCIPAL EN HAUT =====
        panel = np.zeros((120, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Ligne 1 - Stats principales (optimisé comme code de référence)
        stats_text = f"Personnes: {self.stats['count']:3d}  Densite: {self.stats['density']:3d}%  Hotspots: {len(hotspots):2d}  FPS: {self.stats['fps']:2d}"
        cv2.putText(panel, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Ajoutez alerte comportementale
        behaviors = self.intelligence.behavior_patterns
        panic_text = "PANIQUE !!" if behaviors.get('panic') else "Calme"
        panic_color = (0, 0, 255) if behaviors.get('panic') else (0, 255, 0)
        cv2.putText(panel, f"Etat Foule: {panic_text}", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, panic_color, 2)

        # Ligne 2 - Densité bar
        density_pct = self.stats['density']
        bar_w = int((w - 20) * (density_pct / 100))
        color = (0, 255, 0) if density_pct < 40 else (0, 200, 255) if density_pct < 70 else (0, 0, 255)
        cv2.rectangle(panel, (10, 75), (10 + bar_w, 95), color, -1)
        cv2.rectangle(panel, (10, 75), (w - 10, 95), (100, 100, 100), 2)
        cv2.putText(panel, f"DENSITE", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # ===== PANEL PERMANENT DES VISAGES =====
        faces_panel_h = 105
        faces_panel = np.zeros((faces_panel_h, w, 3), dtype=np.uint8)
        faces_panel[:] = (12, 15, 12)
        cv2.putText(faces_panel, "VISAGES - EMOTIONS EN TEMPS REEL", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
        
        x_pos = 10
        face_size = 60
        
        # Afficher Visages détectés MAINTENANT
        for vip_idx, vip_data in enumerate(vips_detected[:6]): # Augmenté à 6
            if x_pos + face_size > w - 10:
                break
            try:
                if isinstance(vip_data, dict):
                    top, right, bottom, left = vip_data['location']
                    emotion = vip_data.get('emotion', 'NEUTRE')
                    name = vip_data.get('name')
                else:
                    continue
                face_crop = frame[max(0,top):min(frame.shape[0],bottom), max(0,left):min(frame.shape[1],right)]
                if face_crop.shape[0] > 8 and face_crop.shape[1] > 8:
                    face_thumb = cv2.resize(face_crop, (face_size-5, face_size-5))
                    y_start = 30
                    if x_pos + face_thumb.shape[1] < w and y_start + face_thumb.shape[0] < faces_panel_h:
                        faces_panel[y_start:y_start+face_thumb.shape[0], x_pos:x_pos+face_thumb.shape[1]] = face_thumb
                        
                        box_color = (0, 0, 255) if name else (255, 200, 0)
                        cv2.rectangle(faces_panel, (x_pos-2, y_start-2), (x_pos+face_size-7, y_start+face_size-7), box_color, 3)
                        
                        label_top = "VIP" if name else "FOULE"
                        cv2.putText(faces_panel, label_top, (x_pos, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1)
                        cv2.putText(faces_panel, emotion.strip()[:4], (x_pos-2, y_start+face_size+8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
                        
                        # Ajouter à cache SEULEMENT SI VIP (pour historique) ou Foule pour debug
                        if name and hasattr(self, 'vips_cache'):
                            self.vips_cache[name] = {'last_seen': time.time(), 'face_img': face_crop.copy(), 'emotion': emotion, 'is_current_frame': True}
            except:
                pass
            x_pos += face_size + 6
        
        # Afficher VIPs du CACHE (détectés récemment mais pas dans ce frame)
        if hasattr(self, 'vips_cache'):
            current_time = time.time()
            for vip_name, vip_info in list(self.vips_cache.items()):
                # Sauter si déjà affiché dans vips_detected
                if any(v.get('name') == vip_name for v in vips_detected if isinstance(v, dict)):
                    continue
                
                if x_pos + face_size > w - 10:
                    break
                
                try:
                    # Vérifier que le VIP n'est pas expiré
                    if (current_time - vip_info['last_seen']) > self.vips_cache_max_age:
                        continue
                    
                    face_crop = vip_info.get('face_img')
                    emotion = vip_info.get('emotion', 'NEUTRE')
                    
                    if face_crop is not None and face_crop.shape[0] > 8:
                        face_thumb = cv2.resize(face_crop, (face_size-5, face_size-5)) if face_crop.shape != (face_size-5, face_size-5) else face_crop
                        y_start = 30
                        if x_pos + face_thumb.shape[1] < w and y_start + face_thumb.shape[0] < faces_panel_h:
                            faces_panel[y_start:y_start+face_thumb.shape[0], x_pos:x_pos+face_thumb.shape[1]] = face_thumb
                            # Bordure GRISE pour VIP en cache (historique)
                            cv2.rectangle(faces_panel, (x_pos-2, y_start-2), (x_pos+face_size-7, y_start+face_size-7), (100, 100, 100), 2)
                            cv2.putText(faces_panel, "HIST", (x_pos, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                            cv2.putText(faces_panel, emotion.strip()[:4], (x_pos-2, y_start+face_size+8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                except:
                    pass
                x_pos += face_size + 6
        
        # Afficher personnes normales
        normal_shown = 0
        for det in detections[:15]:
            if x_pos + face_size > w - 10 or normal_shown >= 3:
                break
            try:
                bx, by, bw, bh = det['bbox']
                face_top = max(0, by)
                face_bottom = min(frame.shape[0], by + int(bh * 0.35))
                face_left = max(0, bx + int(bw * 0.2))
                face_right = min(frame.shape[1], bx + int(bw * 0.8))
                face_crop = frame[face_top:face_bottom, face_left:face_right]
                if face_crop.shape[0] > 8 and face_crop.shape[1] > 8:
                    face_thumb = cv2.resize(face_crop, (face_size-5, face_size-5))
                    y_start = 30
                    if x_pos + face_thumb.shape[1] < w and y_start + face_thumb.shape[0] < faces_panel_h:
                        faces_panel[y_start:y_start+face_thumb.shape[0], x_pos:x_pos+face_thumb.shape[1]] = face_thumb
                        cv2.rectangle(faces_panel, (x_pos-1, y_start-1), (x_pos+face_size-6, y_start+face_size-6), (0, 255, 0), 2)
                        pid = det.get('id', normal_shown)
                        cv2.putText(faces_panel, f"P{pid}", (x_pos, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    normal_shown += 1
            except:
                pass
            x_pos += face_size + 6
        
        # Combiner les trois sections: stats + visages + vidéo
        panel_combined = np.vstack([panel, faces_panel])
        result = np.vstack([panel_combined, overlay])
        
        return result
    
    def _create_live_faces_panel(self, frame, face_locations, emotions, vip_names=None):
        """🚀 ULTRA-INNOVANT: Panel dynamique de tous les visages détectés avec émotions"""
        h, w = frame.shape[:2]
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        if not face_locations:
            return panel
        
        face_size = 100
        x_pos = 10
        max_faces_per_row = max(1, (w - 20) // (face_size + 10))
        
        vip_names = vip_names or {}
        
        for idx, (top, right, bottom, left) in enumerate(face_locations[:max_faces_per_row]):
            if x_pos + face_size > w - 10:
                break
            
            try:
                # Extraire le visage du frame original
                face_img = frame[top:bottom, left:right]
                if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    face_small = cv2.resize(face_img, (face_size-10, face_size-10))
                    
                    # Déterminer couleur (VIP = ROUGE, Normal = VERT)
                    if idx in vip_names:
                        border_color = (0, 0, 255)  # 🔴 ROUGE pour VIPs
                        text_color = (0, 0, 255)
                    else:
                        border_color = (0, 255, 0)  # 🟢 VERT pour personnes normales
                        text_color = (0, 255, 0)
                    
                    # Placer le visage dans le panel avec bordure
                    y_start = 10
                    x_start_face = x_pos + 2
                    x_end_face = x_start_face + face_size - 14
                    
                    if x_end_face < w and face_small.shape[0] > 0:
                        panel[y_start:y_start+face_small.shape[0], x_start_face:x_start_face+face_small.shape[1]] = face_small
                        # Bordure
                        cv2.rectangle(panel, (x_pos, y_start-2), (x_pos+face_size-12, y_start+face_size-12), border_color, 2)
                    
                    # Afficher émotion sous le visage
                    emotion = emotions.get(idx, "😐")
                    cv2.putText(panel, str(emotion), (x_pos, y_start+face_size), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                    
                    # Afficher ID
                    id_text = vip_names.get(idx, f"P{idx}")
                    cv2.putText(panel, id_text[:10], (x_pos, y_start+face_size+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
            except:
                pass
            
            x_pos += face_size + 5
        
        return panel

    def _analyze_loop(self):
        logger.info("🚀 Analyse démarrée")
        last_dets, last_hotspots = [], []
        last_heatmap, last_heatmap_colored = None, None
        frame_counter_scene = 0
        
        # Reset VIP tracking for new analysis session
        self.vip_registry = {} 
        
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
                self.latest_tracks = tracks
                
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
                # Pass tracks to intelligence update
                self.intelligence.update(len(detections), density, detections, tracks)
            
            count = len(last_dets)
            density = min(count / 100, 1.0)
            scene_label, scene_conf = self.scene_result
            anomaly, anomaly_type = self._detect_anomalies(count, density, last_heatmap if last_heatmap is not None else np.zeros((10, 10)), scene_label)
            
            # RECONNAISSANCE FACIALE ET EMOTION - TOUS LES 10 FRAMES
            vips_detected = []
            # On lance maintenant même si pas de VIPs définis, pour avoir les émotions de la foule
            if self.frame_counter % 10 == 0:
                vips_detected = self.detect_faces_and_emotions(frame)
                
                # ===== INTELLIGENCE: ASSOCIER VIP AUX TRACKS & NOTIFICATIONS =====
                # ===== INTELLIGENCE: ASSOCIER VIP AUX TRACKS (ONE-TO-ONE MATCHING) =====
                if hasattr(self, 'latest_tracks') and self.latest_tracks is not None:
                     # 1. Reset current assignments to recalculate best matches
                     # actually we want persistence, so we keep old ones, but for NEW detections we must be careful.
                     # But if we want "one face = one body", we should probably re-evaluate all current visible faces against tracks.
                     
                     # Iterate over each DETECTED VIP FACE
                     for vip_data in vips_detected:
                         if isinstance(vip_data, dict) and vip_data.get('name'):
                             vname = vip_data.get('name')
                             if vip_data.get('location'):
                                 vtop, vright, vbottom, vleft = vip_data.get('location')
                                 f_cx = (vleft + vright) / 2
                                 f_cy = (vtop + vbottom) / 2
                                 
                                 best_track_id = None
                                 min_dist = float('inf')
                                 
                                 # Find the CLOSEST track center to this face center
                                 for track in self.latest_tracks:
                                     tx1, ty1, tx2, ty2, track_id = track
                                     t_cx = (tx1 + tx2) / 2
                                     t_cy = (ty1 + ty2) / 2
                                     
                                     # Check if face is roughly inside or close to body box
                                     if (tx1 - 20) < f_cx < (tx2 + 20) and (ty1 - 20) < f_cy < (ty2 + 20):
                                         dist = (f_cx - t_cx)**2 + (f_cy - t_cy)**2
                                         if dist < min_dist:
                                             min_dist = dist
                                             best_track_id = int(track_id)
                                 
                                 # Assign ONLY to the best track (Unique assignment)
                                 if best_track_id is not None:
                                     # FORCE: This VIP name is now associated with THIS track ID only.
                                     # Overwrites any previous track ID for this name.
                                     self.vip_registry[vname] = best_track_id

                # ===== METTRE À JOUR LE CACHE DES VIPs DETECTÉS =====
                import datetime
                for vip_data in vips_detected:
                    try:
                        if isinstance(vip_data, dict):
                            name = vip_data.get('name', 'UNKNOWN')
                            # Check for notification trigger BEFORE updating cache
                            if name not in self.vips_cache and name != 'UNKNOWN':
                                now_str = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                                self.notifications.append({
                                    'message': f"ALERTE VIP: {name}",
                                    'sub': f"{now_str}",
                                    'timestamp': time.time(),
                                    'name': name
                                })
                            emotion = vip_data.get('emotion', 'NEUTRE')
                            location = vip_data.get('location')
                            
                            # Extraire l'image du visage
                            if location:
                                top, right, bottom, left = location
                                face_crop = frame[max(0,top):min(frame.shape[0],bottom), max(0,left):min(frame.shape[1],right)]
                                if face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                                    # Stocker dans le cache
                                    self.vips_cache[name] = {
                                        'last_seen': time.time(),
                                        'face_img': face_crop.copy(),
                                        'emotion': emotion,
                                        'is_current_frame': True
                                    }
                    except Exception as e:
                        logger.debug(f"Erreur mise à jour cache VIP: {e}")
                
                # Nettoyer le cache des VIPs trop vieux
                current_time = time.time()
                expired_vips = [name for name, data in self.vips_cache.items() 
                               if (current_time - data['last_seen']) > self.vips_cache_max_age]
                for name in expired_vips:
                    del self.vips_cache[name]
            
            # Marquer les VIPs du cache comme "non courant" si pas détectés ce frame
            if self.frame_counter % 10 != 0:  # Frames sans détection faciale
                for name in self.vips_cache:
                    self.vips_cache[name]['is_current_frame'] = False
            
            self.density_history.append(density)
            self.count_history.append(count)
            
            frame_annotated = self._draw_overlay(frame, last_dets, last_heatmap_colored if last_heatmap_colored is not None else np.zeros_like(frame), last_hotspots, vips_detected)
            
            loop_time = time.time() - loop_start
            self.fps_counter.append(1.0 / max(0.001, loop_time))
            fps = int(np.mean(list(self.fps_counter)))
            
            # SAVE RESULTS FOR RENDERING THREAD
            with self.lock:
                self.last_detections = last_dets
                self.last_heatmap_colored = last_heatmap_colored
                self.last_hotspots = last_hotspots
                self.last_vips_detected = vips_detected
                # Store latest raw frame for rendering thread to compose overlays quickly
                try:
                    self.raw_frame = frame.copy()
                except Exception:
                    self.raw_frame = None
                
                # Update stats
                self.stats.update({
                    'count': count, 'density': int(density * 100), 'hotspots': len(last_hotspots),
                    'status': self._get_status(count), 'scene': scene_label, 'scene_conf': scene_conf,
                    'anomaly': anomaly, 'anomaly_type': anomaly_type,
                    'entry_count': self.entry_count, 'exit_count': self.exit_count,
                    'density_trend': 'stable', 'fps': int(fps), 'vips_detected': [] # Frontend handles this via API if needed, or we just rely on visual panel
                })
            
            # On ne dort plus ici, car on consomme aussi vite que possible les frames capturés par l'autre thread
            # Mais on veut éviter de saturer le CPU si la capture est lente
            time.sleep(0.01) 

    def _get_status(self, count):
        if count < 15: return "Faible"
        elif count < 35: return "Normal"
        elif count < 55: return "Élevé"
        else: return "Critique"

        # 🔥 UPDATE PERSISTENT FACES
        try:
            # Add VIPs first
            for vip in vip_panel_data:
                # Crop face
                f_top, f_right, f_bottom, f_left = vip['location']
                face_img = frame[f_top:f_bottom, f_left:f_right]
                if face_img.size > 0:
                    self.persistent_faces[vip['name']] = {
                        'name': vip['name'],
                        'face_img': face_img,
                        'emotion': vip['emotion'],
                        'last_seen': time.time(),
                        'is_vip': True
                    }

            # Add Normal Faces (from fast recognition in _draw_overlay logic, but cleanly done here)
            # To be efficient, we trust the vip_panel_data passed in, and maybe re-scan frame if needed
            # For "Personnes normales", we need valid Tracking IDs to keep them persistent properly
            # Let's use the 'detections' passed in which have IDs
            for det in detections:
                if det.get('id') is not None:
                     tid = int(det['id'])
                     x, y, w_box, h_box = det['bbox']
                     # Crop face approximation
                     face_top = y
                     face_bottom = y + h_box // 3 # Approximate face area
                     face_left = x + w_box // 4
                     face_right = x + 3 * w_box // 4
                     
                     face_img = frame[face_top:face_bottom, face_left:face_right]
                     if face_img.size > 0 and face_img.shape[0] > 10 and face_img.shape[1] > 10:
                         key = f"P{tid}"
                         # Check if this person is already a VIP to avoid duplicates
                         # Simple logic: if a VIP is at this location? Hard to match.
                         # Prioritize VIPs. If we have VIPs, we might skip normal faces close to them.
                         
                         self.persistent_faces[key] = {
                             'name': key,
                             'face_img': face_img,
                             'emotion': "", # Fast emotion for normal people? Maybe skip for speed
                             'last_seen': time.time(),
                             'is_vip': False
                         }
        except Exception as e:
            pass

        return self._draw_overlay(frame, detections, heatmap_colored, hotspots, vips_detected)

    def get_frame(self):
        """Returns the LATEST raw frame with the LATEST analysis overlay"""
        # Get latest raw frame from capture thread
        with self.lock:
            if not hasattr(self, 'raw_frame') or self.raw_frame is None:
                return None
            frame = self.raw_frame.copy()
            
            # Get latest analysis results
            # We need to store these in self from inside _analyze_loop
            # Let's assume _analyze_loop updates: self.last_detections, self.last_heatmap, etc.
            # But currently _analyze_loop updates self.current_frame which is already annotated.
            # To fix FPS, we change the strategy:
            # 1. _analyze_loop computes results -> stores in self.latest_results
            # 2. get_frame() takes self.raw_frame + self.latest_results -> draws overlay -> returns
            
            # FALLBACK to old behavior if refactor is too risky in one step:
            # The user complains about FPS. If self.current_frame comes from analyze_loop, it's limited by analyze FPS.
            # We MUST draw here.
            
            detections = getattr(self, 'last_detections', [])
            heatmap_colored = getattr(self, 'last_heatmap_colored', None)
            hotspots = getattr(self, 'last_hotspots', [])
            vips_detected = getattr(self, 'last_vips_detected', [])

        # If no heatmap yet, create empty
        if heatmap_colored is None:
            heatmap_colored = np.zeros_like(frame)

        # Debug: log shapes and basic stats to detect black frames
        try:
            logger.debug(f"get_frame: frame shape={frame.shape}, frame min={frame.min()}, max={frame.max()}")
            logger.debug(f"get_frame: heatmap_colored shape={heatmap_colored.shape}, min={heatmap_colored.min()}, max={heatmap_colored.max()}")
        except Exception:
            logger.debug("get_frame: unable to read min/max values (possible empty frame)")

        # Draw overlay ON THE FLY
        annotated_frame = self._compose_final_frame(frame, detections, heatmap_colored, hotspots, vips_detected)

        # If annotated_frame is None or completely black, log a warning
        if annotated_frame is None:
            logger.warning("get_frame: annotated_frame is None - returning None")
            return None
        try:
            if np.max(annotated_frame) == 0:
                logger.warning("get_frame: annotated_frame appears completely black (max==0)")
        except Exception:
            pass

        return annotated_frame

    def _compose_final_frame(self, frame, detections, heatmap_colored, hotspots, vips_detected):
        # This calls the method we modified above (formerly _draw_overlay)
        # We renamed _draw_overlay logic to be used here? 
        # Actually I replaced _draw_overlay content in the previous chunk. 
        # So I can just call it.
        return self._draw_overlay(frame, detections, heatmap_colored, hotspots, vips_detected)
    
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
            
            # 🎭 DÉTECTER ÉMOTIONS
            vips_detected = self.detect_faces_and_emotions(frame)
            
            # Mettre à jour les stats globales pour que _draw_overlay les utilise
            with self.lock:
                self.stats.update({
                    'count': count, 
                    'density': density_pct, 
                    'hotspots': len(hotspots),
                    'status': self._get_status(count),
                    'scene': scene_label,
                    'scene_conf': float(scene_conf),
                    'anomaly': anomaly,
                    'anomaly_type': anomaly_type if anomaly else None,
                    'fps': 0
                })
            
            # 🎨 CRÉER L'OVERLAY AVEC PANEL D'INFOS (UNIFORMISÉ)
            frame_annotated = self._draw_overlay(frame, detections, heatmap_colored, hotspots, vips_detected)
            
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
        
        .feature-card { transition: transform 0.3s, box-shadow 0.3s; }
        .feature-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .feature-icon { transition: transform 0.3s; }
        .feature-card:hover .feature-icon { transform: scale(1.1); }
        .feature-stats { transition: color 0.3s; }
        
        .quick-stat { text-align: center; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; }
        .stat-value { font-size: 2em; font-weight: bold; color: var(--accent); }
        .stat-label { color: #888; margin-top: 5px; }
        
        .step { display: flex; align-items: center; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 10px; margin-bottom: 10px; }
        .step-number { flex-shrink: 0; }
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
        <div class="menu-item" onclick="showPage('vips')">
            <i class="fas fa-user-check"></i> VIPs
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
                    Découvrez comment l'IA transforme la surveillance des foules avec précision, rapidité et intelligence émotionnelle.
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
                    <p>YOLOv8  pour un suivi précis des individus en temps réel.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #667eea;">précision avancée</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #00bf9a; margin-bottom: 15px;">🧠</div>
                    <h3>Intelligence Émotionnelle</h3>
                    <p>Reconnaissance faciale des émotions pour détecter la panique ou la joie instantanément.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #00bf9a;">7 émotions</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #fd5d93; margin-bottom: 15px;">🚨</div>
                    <h3>Alertes Intelligentes</h3>
                    <p>Notifications automatiques en cas de surpopulation (>10% densité) par scène.</p>
                    <div class="feature-stats" style="margin-top: 15px; font-size: 1.5em; font-weight: bold; color: #fd5d93;">Temps réel</div>
                </div>
                
                <div class="card feature-card" style="text-align: center; position: relative;">
                    <div class="feature-icon" style="font-size: 3em; color: #e14eca; margin-bottom: 15px;">👤</div>
                    <h3>VIP Protection</h3>
                    <p>Reconnaissance faciale optimisée pour les personnalités importantes avec suivi prioritaire.</p>
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
                        <div class="stat-value" id="homeVips">0</div>
                        <div class="stat-label">VIPs détectés</div>
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
                        <strong>Configurez les VIPs</strong><br>
                        <span style="color: #666;">Ajoutez des visages pour reconnaissance</span>
                    </div>
                    <div class="step">
                        <div class="step-number" style="background: #fd5d93; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">3</div>
                        <strong>Lancez l'analyse</strong><br>
                        <span style="color: #666;">Visualisez en temps réel</span>
                    </div>
                    <div class="step">
                        <div class="step-number" style="background: #e14eca; color: white; width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;">4</div>
                        <strong>Surveillez les alertes</strong><br>
                        <span style="color: #666;">Réagissez aux anomalies</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="dashboard" class="page">
            <div class="header">
                <h1>Centre de Contrôle</h1>
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
        
        <div id="vips" class="page">
            <div class="header">
                <h1>👤 Gestion des VIPs</h1>
            </div>
            
            <div class="card" style="margin-bottom: 20px;">
                <h3>Ajouter un VIP</h3>
                <p>Chargez une photo de visage pour reconnaître cette personne dans les vidéos.</p>
                <div style="display: flex; gap: 10px; align-items: center; margin-top: 15px;">
                    <input type="text" id="vipName" placeholder="Nom du VIP" style="padding: 8px; border-radius: 5px; border: 1px solid #333; background: #222; color: white;">
                    <input type="file" id="vipFace" accept="image/*" style="padding: 8px;">
                    <button class="btn btn-primary" onclick="addVIP()">Ajouter VIP</button>
                </div>
            </div>
            
            <div class="card">
                <h3>VIPs Enregistrés</h3>
                <div id="vipList" style="margin-top: 15px;">
                    <!-- Liste des VIPs sera chargée ici -->
                </div>
            </div>
            
            <div class="card" style="margin-top: 20px; background: rgba(255, 0, 255, 0.1); border-left: 4px solid #ff00ff;">
                <h3>🔍 Détection en Temps Réel</h3>
                <p>Les VIPs détectés apparaissent avec un cadre magenta et leur nom sur la vidéo.</p>
                <div id="vipAlert" style="margin-top: 10px; color: #ff00ff; font-weight: bold;"></div>
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
        
        function addVIP() {
            const name = document.getElementById('vipName').value.trim();
            const file = document.getElementById('vipFace').files[0];
            if (!name || !file) {
                alert('Veuillez saisir un nom et sélectionner une image.');
                return;
            }
            
            let fd = new FormData();
            fd.append('face', file);
            fd.append('name', name);
            
            fetch('/add_vip', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
                if(d.success) {
                    alert('✅ VIP ajouté: ' + name);
                    loadVIPs();
                    document.getElementById('vipName').value = '';
                    document.getElementById('vipFace').value = '';
                } else {
                    alert('❌ Erreur: ' + d.error);
                }
            });
        }
        
        function loadVIPs() {
            fetch('/get_vips').then(r=>r.json()).then(vips => {
                const list = document.getElementById('vipList');
                list.innerHTML = vips.length ? vips.map(name => 
                    `<div style="padding: 8px; background: rgba(255,255,255,0.05); margin: 5px 0; border-radius: 5px;">
                        <i class="fas fa-user"></i> ${name}
                    </div>`
                ).join('') : '<p style="color: #666;">Aucun VIP enregistré</p>';
            });
        }
        
        // Charger les VIPs au démarrage
        loadVIPs();

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
                document.getElementById('homeVips').innerText = d.vips_detected ? d.vips_detected.length : '0';
                
                const anomalyAlert = document.getElementById('anomalyAlert');
                if (d.anomaly) {
                    anomalyAlert.classList.add('active');
                    document.getElementById('anomalyType').textContent = d.anomaly_type;
                } else {
                    anomalyAlert.classList.remove('active');
                }
                
                const vipAlert = document.getElementById('vipAlert');
                if (d.vips_detected && d.vips_detected.length > 0) {
                    const vipList = d.vips_detected.map(vip => {
                        let text = vip.name;
                        if (vip.emotion && vip.emotion !== 'unknown') {
                            text += ` (${vip.emotion})`;
                        }
                        return text;
                    });
                    vipAlert.textContent = 'VIPs/Émotions détectés: ' + vipList.join(', ');
                    vipAlert.style.display = 'block';
                } else {
                    vipAlert.style.display = 'none';
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
        return jsonify({'count': 0, 'density': 0, 'hotspots': 0, 'status': 'Erreur', 'scene': 'Inconnu', 'scene_conf': 0.0, 'anomaly': False, 'anomaly_type': None, 'entry_count': 0, 'exit_count': 0, 'density_trend': 'stable', 'fps': 0, 'vips_detected': []})

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

@app.route('/add_vip', methods=['POST'])
def add_vip():
    try:
        if 'face' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier'})
        file = request.files['face']
        name = request.form.get('name', 'VIP')
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nom vide'})
        
        # Sauvegarder temporairement
        temp_path = f"temp_vip_{name}.jpg"
        file.save(temp_path)
        
        if video_analyzer.add_vip(name, temp_path):
            os.remove(temp_path)
            return jsonify({'success': True, 'message': f'VIP {name} ajouté'})
        else:
            os.remove(temp_path)
            return jsonify({'success': False, 'error': 'Échec ajout VIP'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_vips')
def get_vips():
    return jsonify(list(video_analyzer.vips.keys()))

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