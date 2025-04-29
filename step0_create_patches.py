import argparse
import torch
from torchvision import datasets, transforms

def create_patches(images, num_patches):
    B, C, H, W = images.shape
    patch_size = H // num_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, num_patches * num_patches, patch_size, patch_size)
    return patches

def run_patch_creation(num_patches):
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_images = torch.stack([img for img, _ in mnist_train])
    train_labels = torch.tensor([label for _, label in mnist_train])

    test_images = torch.stack([img for img, _ in mnist_test])
    test_labels = torch.tensor([label for _, label in mnist_test])

    train_patches = create_patches(train_images, num_patches)
    test_patches = create_patches(test_images, num_patches)

    torch.save((train_patches, train_labels), 'train_patches.pt')
    torch.save((test_patches, test_labels), 'test_patches.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_patches', type=int, required=True)
    args = parser.parse_args()
    run_patch_creation(args.num_patches)
