import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        #print(inputs.size())
        targets = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        #print(targets.size())
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        inter = (inputs * targets).sum()
        dice = (2. * inter + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=2).permute(0,3,1,2).float()
        #print(targets.size())
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        #inputs = torch.argmax(inputs, dim=1)
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma*BCE_loss
        return F_loss.mean()



def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler,
                    print_freq=10, min_valid_loss=100):
    train_loss_values = []
    valid_loss_values = []
    for train_iter, pack in enumerate(train_data_loader):
        train_loss = 0
        valid_loss = 0
        img = pack['img'].to(device)
        label = pack['label'].to(device)
        optimizer.zero_grad()
        pred = model(img)
        

        #if pred.size() != label.size():
        #    pred = torch.nn.functional.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)

        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        

        if (train_iter + 1) % print_freq == 0:
            with torch.no_grad():
                model.eval()
                for valid_iter, pack in enumerate(valid_data_loader):
                    img = pack['img'].to(device)
                    label = pack['label'].to(device)
                    pred = model(img)
                    
         #           if pred.size() != label.size():
         #               pred = torch.nn.functional.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)

                    loss = criterion(pred, label)
                    valid_loss += loss.item()

                if min_valid_loss >= valid_loss / len(valid_data_loader):

                    torch.save(model.state_dict(),
                               'best_model_v1.pth')  # set file fname for model save 'best_model_{your name}.pth'

                    min_valid_loss = valid_loss / len(valid_data_loader)
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}' \
                          .format(epoch + 1, train_iter + 1, len(train_data_loader), train_loss / print_freq,
                                  valid_loss / len(valid_data_loader), lr_scheduler.get_last_lr()), \
                          " => model saved")
                    pred_converted = torch.argmax(pred, dim=1)
                    pred_converted = pred_converted[0]
                    pred_converted = pred_converted.cpu().numpy().astype(np.uint8)
                    for i in range(256):
                        for j in range(256):
                            if pred_converted[i][j] == 1:
                                pred_converted[i][j] = 255
#                    print(np.unique(pred_converted))
                    pred_converted = Image.fromarray(pred_converted)
#                    pred_converted.save(f'/home/hojun/results/pred_{train_iter}.png')
                    label_np = label.cpu().numpy().astype(np.uint8)
                    label_np = label_np[0]
                    for i in range(256):
                        for j in range(256):
                            if label_np[i][j] == 1:
                                label_np[i][j] = 255
                    label_np = Image.fromarray(label_np)
 #                   label_np.save(f'/home/hojun/results/label_{train_iter}.png')
                    lr_scheduler.step()
    

                else:
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}' \
                          .format(epoch + 1, train_iter + 1, len(train_data_loader), train_loss / print_freq,
                                  valid_loss / len(valid_data_loader), lr_scheduler.get_last_lr()))
                    
    train_loss_values.append(train_loss / print_freq)
    valid_loss_values.append(valid_loss / len(vaild_data_loader)

    plt.plot(train_loss_values)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
                             
    plt.plot(valid_loss_values)
    plt.title('Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()    
    return min_valid_loss
