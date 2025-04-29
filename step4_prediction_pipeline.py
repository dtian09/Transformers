import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from step1_patch_projection import PatchProjection
from step2_train_test_encoder import TransformerEncoder

def create_patches_single(image, num_patches):
    C, H, W = image.shape
    patch_size = H // num_patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, num_patches * num_patches, patch_size, patch_size)
    patches = patches.unsqueeze(0)
    return patches

def run_prediction(image_path, num_patches=4):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('L')
    image = transform(image)

    patches = create_patches_single(image, num_patches)
    patch_dim = patches.size(-1) * patches.size(-2) * patches.size(1)

    projector = PatchProjection()
    patches_embedded = projector(patches)

    model = TransformerEncoder(embed_dim=64, num_heads=8, num_classes=10)
    model.load_state_dict(torch.load('trained_encoder.pt'))
    model.eval()

    with torch.no_grad():
        pred = model(patches_embedded)
        label = pred.argmax(1).item()
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()
    label = run_prediction(args.image_path)
    print(f'Predicted Label: {label}')
