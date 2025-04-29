import argparse
from step0_create_patches import run_patch_creation
from step1_patch_projection import run_patch_projection
from step2_train_test_encoder import run_train_test_encoder

def main(args):
    run_patch_creation(num_patches=args.num_patches)
    run_patch_projection()
    run_train_test_encoder(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_patches', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
