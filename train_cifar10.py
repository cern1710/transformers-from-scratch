import data_loader as dl
import vision_transformer as ViT
import torch
from torch import nn
import os
from torchvision import transforms

device = torch.device('mps')

def train(model, train_loader, optimizer, criterion, clip_grad_norm: bool = True):
    model.train()
    total_loss, accuracy, total = 0, 0, 0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))
        loss.backward()
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        accuracy += (pred.argmax(dim=-1) == labels).float().sum().item()
        total += labels.numel()

    total_loss /= len(train_loader)
    accuracy /= total
    return total_loss, accuracy

def eval(model, data_loader, criterion):
    model.eval()
    total_loss, accuracy, total = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))

            total_loss += loss.item()
            accuracy += (pred.argmax(dim=-1) == labels).float().sum().item()
            total += labels.numel()

    total_loss /= len(data_loader)
    accuracy /= total
    return total_loss, accuracy

def train_vit(**kwargs):
    model = ViT.VisionTransformer(**kwargs).to(device)
    optimizer, schedulers = model.configure_optimizer()
    optimizer = optimizer[0]
    scheduler = schedulers[0]

    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    best_val_accuracy = 0
    best_model_path = os.path.join("./path", "vit_cifar10.pth")

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader,
                                      optimizer, criterion)
        val_loss, val_acc = eval(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}\n" + "="*40)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = eval(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return model, {'test_acc': test_acc, 'val_acc': best_val_accuracy}

if __name__ == "__main__":
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.49139968, 0.48215827, 0.44653124],
                                            [0.24703223, 0.24348505, 0.26158768])
                                        ])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop((32,32),
                                                                     scale=(0.8, 1.0),
                                                                     ratio=(0.9, 1.1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.49139968, 0.48215827, 0.44653124],
                                            [0.24703223, 0.24348505, 0.26158768])
                                        ])
    train_loader, test_loader, val_loader = dl.load_data(
        val_split=0.1,
        custom_train_transform=train_transform,
        custom_test_transform=test_transform
    )
    assert val_loader is not None

    model, result = train_vit(
        input_dim=256,
        hidden_dim=512,
        num_classes=10,
        num_heads=8,
        num_layers=6,
        num_channels=3,
        patch_size=4,
        num_patches=64,
        dropout=0.1,
        learning_rate=1e-3
    )