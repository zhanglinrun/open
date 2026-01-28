import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import copy

import argparse

def main():
    # 1. Configuration
    parser = argparse.ArgumentParser(description='Train ResNet18 on data_cut_10_v2')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'data_cut_10_v2'), help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    num_classes = 10
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 2. Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. Load Dataset
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    # Split into train and val (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update transform for validation set (strictly speaking random_split shares the underlying dataset, 
    # so we might need a wrapper to apply different transforms, but for simplicity we often just use the same 
    # or rely on the fact that ImageFolder was init with train transforms. 
    # To be "pure" and correct, let's just use train transforms for both or reload.
    # For this simple script, using train transforms (augmentations) on validation is suboptimal but runs.
    # A better way is to create two subsets with different transforms.)
    
    # Correct approach for transforms:
    # Reload dataset for validation to apply correct transforms or use a custom wrapper.
    # Here I will re-instantiate ImageFolder for validation logic to be clean.
    
    train_dataset_full = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    val_dataset_full = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
    
    # We need to ensure we split them continuously/identically. 
    # random_split uses a generator. We can use indices.
    indices = torch.randperm(len(train_dataset_full)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # 4. Model Setup
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # Add Dropout before the final fully connected layer to prevent overfitting
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Increase weight_decay for stronger regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 6. Save Model
    model.load_state_dict(best_model_wts)
    save_path = 'resnet18_data_cut_10_v2.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()