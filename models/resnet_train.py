import torch
import torchvision.models as models
import urllib.request
from collections import OrderedDict
import os

# URLs officielles
URL_MODEL = "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar"
URL_CATEGORIES = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

# Fichiers locaux finaux
MODEL_PTH = "resnet18_places365.pth"
CATEGORIES_FILE = "categories_places365.txt"

# Télécharger le fichier des catégories
if not os.path.exists(CATEGORIES_FILE):
    print(f"Téléchargement de {CATEGORIES_FILE} ...")
    urllib.request.urlretrieve(URL_CATEGORIES, CATEGORIES_FILE)

# Télécharger le checkpoint tar et charger en mémoire directement
print("Téléchargement et chargement du modèle en mémoire...")
checkpoint_data = urllib.request.urlopen(URL_MODEL).read()
import io, torch
checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location=torch.device('cpu'))

# Extraire le state_dict
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # nettoyer DataParallel
    new_state_dict[name] = v

# Créer le modèle ResNet18 pour 365 classes
model = models.resnet18(num_classes=365)
model.load_state_dict(new_state_dict)

# Sauvegarder le modèle final au format .pth
torch.save(model.state_dict(), MODEL_PTH)
print(f"Modèle sauvegardé sous {MODEL_PTH}")
print(f"Fichier des catégories disponible : {CATEGORIES_FILE}")
