import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
import torch.utils.data as data
from torch import nn
import torch.nn.functional as F
from functools import partial

device = torch.device('cpu')

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, accuracy, total = 0, 0, 0

    for batch in train_loader:
        inputs, labels = batch
        inputs = F.one_hot(inputs, num_classes=model.
                           num_classes).float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))
        loss.backward()
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
            inputs = F.one_hot(inputs, num_classes=model.
                               num_classes).float().to(device)
            labels = labels.to(device)

            pred = model(inputs)
            loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))

            total_loss += loss.item()
            accuracy += (pred.argmax(dim=-1) == labels).float().sum().item()
            total += labels.numel()

    total_loss /= len(data_loader)
    accuracy /= total
    return total_loss, accuracy

def train_seq2seq(**kwargs):
    model = ds.ReversePredictor(max_iter=1000, **kwargs).to(device)
    optimizer, schedulers = model.configure_optimizer()
    optimizer = optimizer[0]    # Only 1 optimizer
    scheduler = schedulers[0]['scheduler']

    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    best_val_accuracy = 0
    best_model_path = os.path.join("./path", "best_model.pth")

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
    dataset = partial(ds.ReverseDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128,
                                   shuffle=True, drop_last=True,
                                   pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    inputs, labels = train_loader.dataset[0]
    # print(f"Input data: {inputs}")
    # print(f"Labels: {labels}")

    reverse_model, reverse_result = train_seq2seq(
        input_dim=train_loader.dataset.num_classes,
        model_dim=32,
        num_heads=2,
        num_classes=train_loader.dataset.num_classes,
        num_layers=1,
        dropout=0.0,
        input_dropout=0.0,
        learning_rate=5e-4,
        warmup=50
    )