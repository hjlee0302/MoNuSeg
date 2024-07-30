import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler,
                    print_freq=10, min_valid_loss=100):
    train_loss_list = []
    valid_loss_list = []
    epoch_train_loss = 0
    epoch_valid_loss = 0
    num_batches = len(train_data_loader)
    
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
            with torch.no_grad():
                model.eval()
                for valid_iter, pack in enumerate(valid_data_loader):
                    img = pack['img'].to(device)
                    label = pack['label'].to(device)
                    pred = model(img)
                    
                    #if pred.size() != label.size():
                    #    pred = torch.nn.functional.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)

                    loss = criterion(pred, label)
                    epoch_valid_loss += loss.item()

                avg_valid_loss = epoch_valid_loss / len(valid_data_loader)

                if min_valid_loss >= avg_valid_loss:
                    torch.save(model.state_dict(), 'best_model_v1.pth')
                    min_valid_loss = avg_valid_loss
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}' \
                          .format(epoch + 1, train_iter + 1, len(train_data_loader), epoch_train_loss / (train_iter + 1),
                                  avg_valid_loss, lr_scheduler.get_last_lr()), " => model saved")
                    pred_converted = torch.argmax(pred, dim=1)
                    pred_converted = pred_converted[0]
                    pred_converted = pred_converted.cpu().numpy().astype(np.uint8)
                    pred_converted[pred_converted == 1] = 255
                    pred_converted = Image.fromarray(pred_converted)
                    label_np = label.cpu().numpy().astype(np.uint8)
                    label_np = label_np[0]
                    label_np[label_np == 1] = 255
                    label_np = Image.fromarray(label_np)
                    lr_scheduler.step()
                else:
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}' \
                          .format(epoch + 1, train_iter + 1, len(train_data_loader), epoch_train_loss / (train_iter + 1),
                                  avg_valid_loss, lr_scheduler.get_last_lr()))
                model.train()

    train_loss_list.append(epoch_train_loss / num_batches)
    valid_loss_list.append(avg_valid_loss)

    plt.figure()
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(valid_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_graph.png')
    plt.show()

    return min_valid_loss
