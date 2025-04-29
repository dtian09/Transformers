import argparse
from 0_create_patches import run_patch_creation
from 1_patch_projection import run_patch_projection
from 2_train_test_encoder import run_train_test_encoder

def main(args):
    run_patch_creation(num_patches=args.num_patches)
    run_patch_projection()
    run_train_test_encoder(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_patches', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
