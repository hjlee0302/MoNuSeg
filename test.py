from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
def evaluate(model, test_data_loader, device):
    
    all_preds = []
    all_labels = []
    root = '/home/hojun/MoNuSeg/MoNuSeg_results/'
    transform_to_pil = transforms.ToPILImage()
    model.eval()
    for iter, pack in enumerate(test_data_loader):
        img = pack['img'].to(device)
        label = pack['label'].to(device)
        pred = model(img)
        pred = torch.argmax(pred, dim=1)
        
        if iter < 20:
          
            image = transform_to_pil(img[0])
            image.save(root + f'img_{iter}.png')
            prediction = transform_to_pil(pred[0].float())
            prediction.save(root + f'pred_{iter}.png')
            mask = transform_to_pil(label[0].float())
            mask.save(root + f'label_{iter}.png')

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print('accuracy: {} recall: {} precision: {} f1 score: {}'.format(accuracy,
                                                                          recall,
                                                                          precision,
                                                                          f1))
