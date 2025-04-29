import argparse
from step0_create_patches import run_patch_creation
from step1_patch_projection import run_patch_projection
from step2_train_test_integration_of_encoder_and_mlp import run_train_test_integration_of_encoder_and_mlp

def main(args):
    run_patch_creation(num_patches=args.num_patches)
    run_patch_projection()
    run_train_test_integration_of_encoder_and_mlp(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, patience=args.patience, dropout=args.dropout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_patches', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
