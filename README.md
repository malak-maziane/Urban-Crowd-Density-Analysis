# 🚀 SADU PRO — Analyse Intelligente de Densité Urbaine

SADU PRO (*Système d'Analyse de Densité Urbaine*) est un système complet, optimisé et de pointe conçu pour l'analyse en temps réel des flux de population dans des environnements urbains complexes. Qu'il s'agisse de vidéos de vidéosurveillance (CCTV), de flux de drones (fichiers MP4), ou d'images statiques (JPG/PNG), SADU PRO fournit une couverture complète de la segmentation de la foule, du comptage précis, et de l'analyse comportementale ultra-rapide.

Ce système professionnel a été développé pour **mesurer l'occupation d'une rue, d'une plage, d'un marché ou d'une manifestation**, **estimer la densité de personnes** avec une grande précision, et **détecter les zones fortement peuplées** afin d'optimiser la sécurité publique, la gestion du trafic et anticiper les mouvements de panique ou les besoins d'évacuation.

---

## 🛠 Deux Interfaces Puissantes : `crowdinit.py` & `crowdinnov.py`

Le système propose deux cœurs d'exécution principaux. **Les deux scripts sont entièrement compatibles et fonctionnent parfaitement pour analyser tant des vidéos (MP4) que des images fixes.** 

1. **`crowdinit.py` (L'Édition Standard Maximisée)** : 
   L'application principale qui lance le pipeline complet de nettoyage d'image, de détection de scène, de segmentation de foule, de tracking individuel, et de génération de "Heatmaps" (cartes de chaleur de densité). L'outil indispensable et épuré pour tout contrôle d'une infrastructure.
   
2. **`crowdinnov.py` (L'Édition Innovation "Innov")** : 
   Véritable bijou d'intelligence augmentée, cette version inclut toutes les fonctionnalités de la version standard, auxquelles s'ajoutent des **super-pouvoirs biométriques** uniques :
   - **Détection VIP & Reconnaissance Faciale** : Permet à l'opérateur d'ajouter dynamiquement le visage d'une personne cible. Le système isolera, identifiera et traquera spécifiquement cet individu dans une foule gigantesque grâce à des descripteurs HOG ultra-rapides.
   - **Analyse d'Émotions Multi-Agents** : Détecte instantanément le visage de LA cible VIP, mais extrapole aussi pour analyser de manière transparente **les émotions de tous les autres individus** capturés par la caméra (Neutre, Joie, Colère, Peur, etc.). Cette version dresse un véritable profil psychologique et émotionnel instantané de l'état d'esprit de la foule.

---

## ✨ Fonctionnalités Uniques & Déploiement Technologique

L'application marie l'état de l'art du Deep Learning avec une optimisation algorithmique en C++ via **PyTorch** et OpenCV. PyTorch a été choisi pour sa souplesse, son optimisation GPU CUDA de classe mondiale, et sa capacité à faire collaborer simultanément jusqu'à cinq réseaux de neurones complexes sans impacter les FPS.

### 1. 🚨 Gestion de Capacité & Alarmes de Danger (Crowd Crush Prevention)
L'outil ne se limite pas à compter, il protège. Le système calcule dynamiquement le **pourcentage de remplissage d'un lieu**. Si la jauge d'occupation estimée (le volume d'individus) **dépasse une capacité ou un pourcentage limite défini par l'administrateur**, une **alarme de sécurité visuelle et sonore est automatiquement déclenchée**. 
*Enjeux : Idéal pour prévenir et sauver des vies face aux bousculades urbaines (crowd crushes) ou aux engorgements de trafic.*

### 2. 🌍 Détection Contextuelle de Lieux (Scene Recognition)
Le modèle identifie automatiquement et en millisecondes le type d'endroit analysé (ex: *Plage, Marché, Rue, Parc*). Cette ingénierie permet d’adapter statistiquement ce qu'est un seuil de risque. Par conséquent, une foule dense acceptée dans un stade ouvert sera classifiée différemment s'il s'agit du couloir d'un centre commercial fermé !

### 3. 🧬 Modèles d'Intelligence Artificielle & Entraînements Stratégiques
Nos modèles ont été rigoureusement sélectionnés et pré-entraînés sur des bases de données de recherche mondiales pour garantir la plus haute fidélité possible :

- **CSRNet (Congested Scene Recognition Network)** : Le cerveau de la densité de foule.
  - *Scripts & Poids :* Dossier `models/csrnet_training_system.py` et poids `models/csrnet_best.pth`.
  - *Dataset :* **ShanghaiTech**. Ce dataset est la référence mondiale absolue et académique en matière de surveillance de foule dense. Y entraîner notre CSRNet garantit que même lors d'un concert de 50 000 personnes où de nombreuses caméras ne verraient qu'une "masse noire", notre modèle repère et identifie la fraction anatomique de chaque tête pour un rendu heatmap d'une précision chirurgicale.
- **U-Net (Encodeur ResNet34 pour Segmentation au Pixel)** : 
  - *Scripts & Poids :* `models/unet_train.py` (dossier interne : `models/people_segmentation/`) / poids : `unet_best.pth`.
  - *Pourquoi ce choix architectural ?* La puissance de l'architecture en U combinée à PyTorch offre une détection par segmentation. Contrairement à de simples encadrés autour des gens, il dessine littéralement leurs silhouettes pour extraire les individus des éléments de background de la rue avec l'exigence d'une chirurgie reconstructive.
- **YOLOv8 & DeepSort / Kalman Filters (Tracking Révolutionnaire)** : 
  - Utilisé en tandem pour le comptage individuel unique. Un individu allant faire une course et revenant cinq minutes après dans le champ de la caméra ne sera pas compté deux fois par le système de flux grâce à la prédiction trajectoire du SortTracking System.

### 4. 🤔 Intelligence Prédictive & Analyse Comportementale
- **Moteur d'Intelligence (Intelligence Engine)** : Calcule la vélocité et les trajectoires pour trier l'état vectoriel d'une personne : Stationnaire, Marche, Course.
- **Détection de Mouvements de Panique** : Si plus de 30% d'une foule pourtant immobile se met de façon synchronisée à courir, le système détecte une **anomalie systémique** (mouvement de panique) et bascule son niveau de risque du statut "Modéré" à "CRITIQUE - PANIQUE", appelant immédiatement l'action humaine.

---

## 🚀 Guide de Démarrage (Déploiement Local)

**1. Installation des dépendances complètes**
```bash
python -m pip install -r requirements.txt
```

**2. Démarrage de l'analyseur Flask Web Platform**
```bash
# Pour lancer la version Haute Précision :
python crowdinit.py

# Pour lancer la version avec détection VIP, FaceID, et Émotions de la foule :
python crowdinnov.py
```
Ouvrir votre navigateur sur [http://localhost:5000](http://localhost:5000) et la magie opère en réseau local !

---

## 📜 Notice de Licence d'Utilisation

**Licence MIT (Open Source Spécifique)**

*(Consultez le fichier `LICENSE` pour l'entièreté des conditions légales).*

Copyright (c) 2026 SADU PRO Contributors

L'autorisation est accordée, à titre gratuit, à toute personne obtenant une copie de ce logiciel et des fichiers de documentation associés, de traiter le Logiciel sans restriction, y compris, sans s'y limiter, les droits d'utiliser, de copier, de modifier, de fusionner, de publier, de distribuer, de sous-licencier, et/ou de vendre des copies du Logiciel, sous la condition de toujours y associer la notice de copyright.

> **💡 Avertissement Législatif et Politique de Confidentialité :**
> SADU PRO incorpore des technologies de détection biométrique avancée (reconnaissance faciale, expressions émotionnelles). Lors d'un déploiement avec des caméras réelles orientées vers une rue publique ou une entreprise, **l'opérateur déploie ceci sous sa propre responsabilité légale**. Vous devez vous assurer de votre strict respect des lois internationales et locales sur la protection des données (telles que le **RGPD** en Europe), du consentement à l'image, et de la déclaration de vidéosurveillance aux autorités compétentes de votre pays. Le respect de la vie privée reste une brique indissociable d'une IA éthique et robuste.