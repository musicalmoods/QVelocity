import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

trainingdata = [
    r'Training_Data\Track1\Edge_Case',
    r'Training_Data\Track1\Normal_Driving',
    r'Training_Data\Track2\Edge_Cases',
    r'Training_Data\Track2\Normal_Driving_Data'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vision_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, 4) # 4 features for 4 qubits
vision_model.eval().to(device)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, 60, 320, 140))), # Crop out unnecessary stuff (sky, cars hood, etc)
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

latent_data = []
CORRECTION = 0.2

for folder in trainingdata:
    csv_path = os.path.join(folder, 'driving_log.csv')
    if not os.path.exists(csv_path):
        print(f"Skipping {folder}: No driving_log.csv found.")
        continue

    df = pd.read_csv(csv_path, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    for index, row in df.iterrows():
        steering_center = float(row['steering'])
        cameras = {
            'center': (os.path.join(folder, 'IMG', row['center'].split('\\')[-1].split('/')[-1]), steering_center),
            'left': (os.path.join(folder, 'IMG', row['left'].split('\\')[-1].split('/')[-1]), steering_center + CORRECTION),
            'right': (os.path.join(folder, 'IMG', row['right'].split('\\')[-1].split('/')[-1]), steering_center - CORRECTION)
        }

        for cam_type, (img_path, steering) in cameras.items():
            try:
                image = Image.open(img_path)
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = vision_model(img_tensor).cpu().numpy()[0]
                latent_data.append([*features, steering])
                
                image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                img_tensor_flipped = transform(image_flipped).unsqueeze(0).to(device)
                steering_flipped = -steering
                
                with torch.no_grad():
                    features_flipped = vision_model(img_tensor_flipped).cpu().numpy()[0]
                latent_data.append([*features_flipped, steering_flipped])
                
            except Exception as e:
                pass

out_df = pd.DataFrame(latent_data, columns=['f1', 'f2', 'f3', 'f4', 'steering'])
out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
out_df.to_csv('quantum_training_data_master.csv', index=False)
print(f"Extraction complete! Saved {len(out_df)} augmented samples to quantum_training_data_master.csv")
