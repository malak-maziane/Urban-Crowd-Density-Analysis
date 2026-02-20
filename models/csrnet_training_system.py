# =========================================================
# CSRNet Training Script
# Dataset: ShanghaiTech
# =========================================================

import os
import h5py
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# -------------------------------
# 1) Définir CSRNet (simplifié)
# -------------------------------
def CSRNet():
    input_img = Input(shape=(None, None, 3))

    # Frontend VGG16
    x = Conv2D(64, 3, padding='same', activation='relu')(input_img)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)

    # Backend dilated conv
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)

    # Output layer: 1 channel density map
    output = Conv2D(1, 1, padding='same', activation='linear')(x)

    model = Model(inputs=input_img, outputs=output)
    return model

# -------------------------------
# 2) Préparation des données
# -------------------------------
def load_image_and_gt(image_path, gt_path):
    # Charger image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Charger ground truth density map
    with h5py.File(gt_path, 'r') as f:
        gt = np.array(f['density'])

    # Ajuster dimensions
    if len(gt.shape) == 2:
        gt = np.expand_dims(gt, axis=-1)

    return img, gt

def data_generator(image_dir, gt_dir, batch_size=1):
    images = sorted([os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    gts = sorted([os.path.join(gt_dir,f) for f in os.listdir(gt_dir) if f.endswith('.h5')])
    n = len(images)
    while True:
        for i in range(0, n, batch_size):
            batch_imgs, batch_gts = [], []
            for j in range(i, min(i+batch_size, n)):
                img, gt = load_image_and_gt(images[j], gts[j])
                batch_imgs.append(img)
                batch_gts.append(gt)
            yield np.array(batch_imgs), np.array(batch_gts)

# -------------------------------
# 3) Paramètres et chemins
# -------------------------------
dataset_path = 'dataset/ShanghaiTech'
train_image_dir = os.path.join(dataset_path, 'part_A_final/train_data/images')
train_gt_dir = os.path.join(dataset_path, 'part_A_final/train_data/ground_truth')
val_image_dir = os.path.join(dataset_path, 'part_A_final/test_data/images')
val_gt_dir = os.path.join(dataset_path, 'part_A_final/test_data/ground_truth')

batch_size = 1
epochs = 100

# -------------------------------
# 4) Initialiser le modèle
# -------------------------------
model = CSRNet()
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')

# -------------------------------
# 5) Callbacks
# -------------------------------
checkpoint = ModelCheckpoint('csrnet_trained.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# -------------------------------
# 6) Entraînement
# -------------------------------
train_gen = data_generator(train_image_dir, train_gt_dir, batch_size)
val_gen = data_generator(val_image_dir, val_gt_dir, batch_size)

train_steps = len(os.listdir(train_image_dir)) // batch_size
val_steps = len(os.listdir(val_image_dir)) // batch_size

model.fit(train_gen,
          steps_per_epoch=train_steps,
          validation_data=val_gen,
          validation_steps=val_steps,
          epochs=epochs,
          callbacks=[checkpoint, reduce_lr])
