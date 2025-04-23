import dataset_loader
import utils
import os
from tqdm import tqdm
import wandb

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch.optim as optim

def do_pca(X, n_components=100):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x


if __name__ == '__main__':

    wandb.init(
        project="picollage",
        name="correlation_assignment",
        config={
            "lr": 0.001,
            "epochs": 50,
            "batch_size": 128,
            "weight_decay": 1e-4,
            "pca_components": 100,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    csv_path = os.path.expanduser("~/dataset/correlation_assignment/responses.csv")
    img_dir = os.path.expanduser("~/dataset/correlation_assignment/images")
    dataset = dataset_loader.CorrelationDataset(csv_path, img_dir, transform)

    X, y = utils.extract_features(dataset)
    print("start")
    X_pca = do_pca(X)
    print("---------------------------finish pca----------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)


    model = MLP(in_dim=100).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in tqdm(range(50), desc="Training"):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
        tqdm.write(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

        model.eval()
        total_test_loss = 0.0
        preds_list, trues_list = [], []
    
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                total_test_loss += loss.item()
                preds_list.append(preds.cpu())
                trues_list.append(y_batch.cpu())
    
        test_preds = torch.cat(preds_list).numpy()
        test_trues = torch.cat(trues_list).numpy()
        test_mse = ((test_preds - test_trues)**2).mean()
        wandb.log({"test_mse": test_mse})
        print(f"Test MSE: {test_mse:.4f}")
        
    torch.save(model.state_dict(), "model.pth")
    wandb.finish()
