import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_layers=6, max_len=16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': MLP(embed_dim)
            }) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            x_norm = layer['norm2'](x)
            mlp_out = layer['mlp'](x_norm)
            x = x + mlp_out
        x = x.mean(dim=1)
        return self.fc(x)

def run_train_test_encoder(batch_size=64, epochs=5, lr=1e-3):
    train_embeds, train_labels = torch.load('train_embeds.pt')
    test_embeds, test_labels = torch.load('test_embeds.pt')

    train_dataset = TensorDataset(train_embeds, train_labels)
    test_dataset = TensorDataset(test_embeds, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerEncoder(embed_dim=train_embeds.shape[-1], num_heads=8, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

    print(f'Test Accuracy: {correct / total:.4f}')
    torch.save(model.state_dict(), 'trained_encoder.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    run_train_test_encoder(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
