import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, num_epochs, lr_scheduler,
                    print_freq=10, min_valid_loss=100):
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_valid_loss = 0
        model.train()  # Ensure the model is in training mode

        for train_iter, pack in enumerate(train_data_loader):
            img = pack['img'].to(device)
            label = pack['label'].to(device)
            optimizer.zero_grad()
            pred = model(img)

            #if pred.size() != label.size():
            #    pred = torch.nn.functional.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if (train_iter + 1) % print_freq == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{train_iter+1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

        train_loss_list.append(epoch_train_loss / len(train_data_loader))

        with torch.no_grad():
            model.eval()  # Ensure the model is in evaluation mode
            for valid_iter, pack in enumerate(valid_data_loader):
                img = pack['img'].to(device)
                label = pack['label'].to(device)
                pred = model(img)
                
                #if pred.size() != label.size():
                #    pred = torch.nn.functional.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)

                loss = criterion(pred, label)
                epoch_valid_loss += loss.item()

            avg_valid_loss = epoch_valid_loss / len(valid_data_loader)
            valid_loss_list.append(avg_valid_loss)

            if min_valid_loss >= avg_valid_loss:
                torch.save(model.state_dict(), 'best_model_v1.pth')
                min_valid_loss = avg_valid_loss
                print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_valid_loss:.4f}, Model saved!')

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss_list[-1]:.4f}, Validation Loss: {avg_valid_loss:.4f}')

            lr_scheduler.step()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), valid_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_graph.png')
    plt.show()

    return min_valid_loss
