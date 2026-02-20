# 🚀 SADU PRO — Analyse Intelligente de Densité Urbaine

SADU PRO (*Système d'Analyse de Densité Urbaine*) est un système complet, optimisé et de pointe conçu pour l'analyse en temps réel des flux de population dans des environnements urbains complexes. Qu'il s'agisse de vidéos de vidéosurveillance (CCTV) ou de flux de drones, SADU PRO fournit une couverture complète de la segmentation de la foule, du comptage précis, et de la détection comportementale avancée.

Ce projet capital a été développé pour **mesurer l'occupation d'une rue, d'une plage, d'un marché ou d'une manifestation**, **estimer la densité de personnes** avec une grande précision, et **détecter les zones fortement peuplées** pour optimiser la sécurité publique, le trafic, et anticiper les mouvements de panique ou les besoins d'évacuation.

---

## ✨ Fonctionnalités Principales & Technologies Déployées

L'application marie l'état de l'art du Deep Learning avec une optimisation algorithmique pour fonctionner en temps réel de manière fluide. Par l'utilisation de multiples réseaux de neurones s'exécutant en synergie, l'application garantit une précision remarquable.

### 1. 🧬 Segmentation Sémantique de Foule (U-Net & Fallback)
*Pour détourer avec précision la présence humaine dans des environnements variés.*
- **U-Net avec encodeur ResNet34** : Un modèle ultra-précis développé pour segmenter finement les individus. L'architecture encodeur-décodeur permet d'extraire les features spatiales profondes.
- **YOLOv8-Seg (Fallback)** / **HRNet / DeepLabv3** : L'intégration d'architectures de segmentation par instance (YOLOv8-seg) est implémentée en fallback pour garantir qu'aucune frame critique ne soit perdue si le modèle principal nécessite un relais de performance.

### 2. 📊 Comptage de Foule & Cartographie de Densité
*Fournit le nombre exact ou approximatif d'individus et identifie visuellement les "Points Chauds" (Hotspots).*
- **CSRNet (Congested Scene Recognition Network)** : Le cœur du moteur d'estimation de densité via des réseaux convolutifs dilatés. Il génère des "Density Maps" (cartes de densité) de haute qualité, où les zones rouges indiquent les plus fortes densités. Parfait pour les foules extrêmement denses impossibles à compter individuellement.
- **YOLOv8n + DeepSort / KalmanBoxTracker** : Utilisé pour la détection pure et le tracking (suivi) d'individus uniques sur la vidéo, permettant de calculer les flux réels (E/S) sans double comptage.

### 3. 🤔 Intelligence Prédictive & Analyse Comportementale
*Le système ne se contente pas de voir, il comprend la dynamique de la foule.*
- **Moteur d'Intelligence (Intelligence Engine)** : Calcule la vélocité et les trajectoires (Stationnaire, Marche, Course).
- **Détection d'Anomalies (Panique / Évacuation)** : Si plus de 30% de la foule traquée se met soudainement à courir (algorithme de vitesse basé sur les identifiants uniques de tracking), un risque "CRITIQUE - PANIQUE" est déclenché.
- **Prédictions à 5 et 15 minutes** : En se basant sur la dérivée de l'accumulation ("Trend Factor"), le système anticipe l'évolution de la foule.

### 4. 🎭 Reconnaissance Faciale & Détection d'Émotions
- **HOG Facial Recognition** : Pipeline ultra-rapide capable de reconnaître des VIPs d'après une base de données locale d'encodages biométriques.
- **Analyse d'Émotions (Fast Emotion Analytics)** : Détermination instantanée de la rougeur cutanée et des micro-contrastes (Laplacian variance) pour inférer si les individus (VIPs ou foule classique) sont Neutres, En Colère, Stressés ou Surpris.

### 5. 🌍 Détection de Scène Contextuelle (Places365)
- **Modèle ResNet18 Places365** : Analyse l'arrière-plan de la vidéo (ex: "Plaza", "Park", "Market") pour contextualiser les alertes de foule selon l'environnement.

---

## 🛠 Architecture du Code (`crowdinnov.py` & `crowdinit.py`)

L'application repose sur un écosystème Python robuste alliant un backend PyTorch pour les modèles lourds et Flask pour l'interface de contrôle.

- `UltraFastVideoAnalyzer` : Le chef d'orchestre thread-safe qui ingère le flux vidéo, gère les queues asynchrones, et dispatche les frames vers les sous-modèles (Scene, VIP, Density, Trackers) sans bloquer les FPS.
- `IntelligenceEngine` : Tient un registre des identifiants SORT (Simple Online and Realtime Tracking) pour mémoriser l'historique de déplacement de chaque individu sur plusieurs secondes, rendant possible l'extrapolation des comportements spatio-temporels.
- `SortTracker` & `KalmanBoxTracker` : Implémentation customisée du célèbre tracking SORT, optimisée en numpy pour prédire les bounding boxes avec le filtre de Kalman et associer les détections d'une frame à l'autre via la matrice Munkres / Hungarian Algorithm.
- `Flask Dashboard` : Expose des endpoints REST (`/get_stats`, `/upload_video`) mis à jour asynchrone pour afficher la Heatmap en base64 instantanément sur le navigateur.

---

## 🚀 Lancement Rapide (Local)

**1. Installation des dépendances**
```bash
python -m pip install -r requirements.txt
```

**2. Démarrage de l'analyseur**
```bash
python crowdinnov.py
# Ou
python crowdinit.py
```
Ouvrir votre navigateur sur [http://localhost:5000](http://localhost:5000).

---

## 🔒 Licence et Données
Propulsé avec ❤️ par la technologie d'Intelligence Artificielle.
Distribué sous la licence **MIT**. Veuillez consulter le fichier `LICENSE` pour plus de détails. En cas d'utilisation publique capturant des visages, veillez à vous conformer aux réglementations de confidentialité locales (ex: RGPD).