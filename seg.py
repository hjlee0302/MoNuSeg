import torch
from train import train_one_epoch
from Model import UNet
import torch.nn as nn
import torch.nn.functional as F
from Datasets import BratsDataset_seg
from test import evaluate
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from train.py import DiceLoss, FocalLoss
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # gpu가 사용가능하다면 gpu를 사용하고, 아니라면 cpu를 사용함
print(device)
## Hyper-parameters
num_epochs =30

model_channel = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
model_channel.to(device)

optimizer = torch.optim.Adam(model_channel.parameters(), lr=0.0005)#weight_decay=0.0001
#criterion = DiceLoss().cuda()
#criterion = FocalLoss().cuda()
class_weights = torch.tensor([0.5, 0.5])
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
#criterion  = DiceLoss().to(device)
#criterion = FocalLoss().to(device)
# step_size 이후 learning rate에 gamma만큼을 곱해줌 ex) 111번 스텝 뒤에 lr에 gamma를 곱해줌
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=111,
                                               gamma=1) #0.9 ## learning rate decay

## data loader
train_dataset = BratsDataset_seg('/media/NAS/nas_32/hojun/MoNuSeg_data')
valid_dataset = BratsDataset_seg('/media/NAS/nas_32/hojun/MoNuSeg_data', 'val_patch')
test_dataset = BratsDataset_seg('/media/NAS/nas_32/hojun/MoNuSeg_data', 'test_patch')


train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=4)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, shuffle=True, num_workers=4)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, num_workers=4)

train_loss_list = []
valid_loss_list = []
min_val_loss = 100
for epoch in range(num_epochs):
    train_loss, val_loss, min_val_loss = train_one_epoch(model_channel, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler, print_freq=10, min_valid_loss=min_val_loss)
   # print('validation...')
    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
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
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
model.to(device)
model.load_state_dict(torch.load('best_model_v1.pth'))

evaluate(model, valid_data_loader, device=device)
