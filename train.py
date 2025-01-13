from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import torch

from data import COCODataset
from model import VisionLanguageModel, VisionLanguageLoss

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

def train_vlm(data_dir, vision_model, text_model, epochs=1, batch_size=16, lr=5e-5):
    with torch.no_grad():
        torch.cuda.empty_cache()

    dataset = COCODataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: x)

    model = VisionLanguageModel(vision_model, text_model)
    criterion = VisionLanguageLoss(text_model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    batch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

        for batch in progress_bar:
            images, captions = zip(*batch)
            images = torch.stack(images).to(device)

            outputs, tokenized_captions, base_attn_dim = model(images, captions)

            loss = criterion(outputs, tokenized_captions, base_attn_dim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'vlm.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.close()
