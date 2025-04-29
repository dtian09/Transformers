import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_layers=6, max_len=16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'linear': nn.Linear(embed_dim, embed_dim)
            }) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x_residual = x
            x = layer['norm1'](x)
            attn_out, _ = layer['attn'](x, x, x)
            attn_out = self.dropout(attn_out)
            x = x_residual + attn_out  # Residual connection after attention

            x_residual = x
            x = layer['norm2'](x)
            mlp_out = layer['linear'](x)
            mlp_out = self.dropout(mlp_out)
            x = x_residual + mlp_out  # Residual connection after MLP

        x = x.mean(dim=1)
        return x

def run_train_test_integration_of_encoder_and_mlp(batch_size=64, epochs=5, lr=1e-3, patience=3, dropout=0.1):
    train_embeds, train_labels = torch.load('train_embeds.pt')
    test_embeds, test_labels = torch.load('test_embeds.pt')

    train_dataset = TensorDataset(train_embeds, train_labels)
    test_dataset = TensorDataset(test_embeds, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    wandb.init(project='mnist-transformer', config={
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'early_stopping_patience': patience,
        'dropout': dropout
    })

    encoder = TransformerEncoder(embed_dim=train_embeds.shape[-1], num_heads=8, num_classes=10, max_len=train_embeds.shape[1])
    mlp_classifier = nn.Sequential(
        nn.Linear(train_embeds.shape[-1], 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    encoder.dropout.p = dropout
    params = list(encoder.parameters()) + list(mlp_classifier.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        encoder.train()
        mlp_classifier.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            optimizer.zero_grad()
            features = encoder(xb)
            preds = mlp_classifier(features)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        encoder.eval()
        mlp_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                features = encoder(xb)
                preds = mlp_classifier(features)
                correct += (preds.argmax(1) == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(encoder, 'trained_encoder.pt')
            torch.save(mlp_classifier, 'trained_mlp.pt')
            wandb.save('trained_encoder.pt')
            wandb.save('trained_mlp.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    encoder = torch.load('trained_encoder.pt')
    mlp_classifier = torch.load('trained_mlp.pt')
    encoder.eval()
    mlp_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            features = encoder(xb)
            preds = mlp_classifier(features)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f}')
    wandb.log({"test_accuracy": test_acc})
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    run_train_test_integration_of_encoder_and_mlp(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, dropout=args.dropout)

