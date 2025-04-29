import argparse
import torch
import torch.nn as nn

class PatchProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(7*7, 64)

    def forward(self, patches):
        B, C, NP, H, W = patches.shape
        patches = patches.view(B, NP, -1)  # Flatten each patch
        embeddings = self.proj(patches)
        return embeddings

def run_patch_projection():
    train_patches, train_labels = torch.load('train_patches.pt')
    test_patches, test_labels = torch.load('test_patches.pt')

    model = PatchProjection()
    train_embeds = model(train_patches)
    test_embeds = model(test_patches)

    torch.save((train_embeds, train_labels), 'train_embeds.pt')
    torch.save((test_embeds, test_labels), 'test_embeds.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_patch_projection()
