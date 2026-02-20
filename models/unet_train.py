import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# ===========================
# 1️⃣ Dataset rapide
# ===========================
class PeopleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = (mask > 0.5).float()  # masque 0/1
        
        return image, mask

# ===========================
# 2️⃣ Transformations rapides
# ===========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = PeopleSegmentationDataset(
    image_dir="people_segmentation/images",
    mask_dir="people_segmentation/masks",
    transform=transform
)

# Subset pour accélérer
subset_size = 500
dataset = Subset(dataset, list(range(subset_size)))

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"Nombre d'images dans le dataset subset: {len(dataset)}")
for images, masks in dataloader:
    print(f"Batch images: {images.shape}, Batch masks: {masks.shape}")
    break

# ===========================
# 3️⃣ UNet simple
# ===========================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        
        self.out = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = torch.sigmoid(self.out(d1))
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# ===========================
# 4️⃣ Optimizer et loss
# ===========================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===========================
# 5️⃣ Entraînement rapide
# ===========================
num_epochs = 5
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        if (i+1) % 10 == 0:
            print(f"Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss moyenne: {epoch_loss:.4f}")
    
    # sauvegarde du meilleur modèle
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "unet_best.pth")
        print("💾 Meilleur modèle sauvegardé !")
