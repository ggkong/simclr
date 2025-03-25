import torch
from torch.utils.data import DataLoader
from aptos_dataset import APTOSSimCLRDataset
from simclr_model import SimCLRModel
from contrastive_loss import nt_xent_loss
import os
import torch
torch.cuda.empty_cache()

def main():
    device = 'cuda'

    dataset = APTOSSimCLRDataset(image_dir='train_images')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 50
    save_interval = 1  # 每隔10个epoch保存一次模型

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for xi, xj in loader:
            xi, xj = xi.to(device), xj.to(device)

            zi = model(xi)
            zj = model(xj)
            loss = nt_xent_loss(zi, zj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {epoch_loss:.4f}")

        if epoch % save_interval == 0 or epoch == num_epochs:
            save_path = f'checkpoints/simclr_epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至 {save_path}")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
