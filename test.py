from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

def evaluate(model, test_data_loader, device):
    accuracy = 0

    iou_list = [0, 0, 0]

    for iter, pack in enumerate(test_data_loader):
        img = pack['img'].to(device)
        label = pack['label']
        pred = model(img)
        pred = pred.cpu().detach().numpy()
        (B, C, W, H) = pred.shape
        pred = np.argmax(pred[0], axis=0)

        for num_channel in range(C):
            count_pred = np.sum(pred == num_channel)
            count_label = np.sum(label[0].numpy() == num_channel)
            intersection = np.sum(label[0].numpy() * 4 == pred + (3 * num_channel))
            iou = intersection / (count_pred + count_label - intersection + 1e-10)
            iou_list[num_channel] += iou

                # plt.imshow(pred==num_channel)
                # plt.show()
                # plt.imshow(label[0].numpy()==num_channel)
                # plt.show()
                # plt.imshow(label[0].numpy()*4 == pred+(3*num_channel))
                # plt.show()
                # print(count_pred, count_label, intersection, iou)
        ious = []
        for channel in range(len(iou_list)):
            zero_index = np.where(iou_list[channel] == 0)
            miou = np.delete(iou_list[channel], zero_index)
            print('{} channel miou: {}'.format(channel, miou / len(test_data_loader)))
            ious.append(miou / len(test_data_loader))
        print(f"average iou : {np.mean(ious[1:])}")

    accuracy = 0
    pred_list = []
    label_list = []
    for iter, pack in enumerate(test_data_loader):
        img = pack['img'].to(device)
        label = pack['label']
        pred = model(img)
        pred = pred.cpu().detach().numpy()
        label_list.append(label)

        if pred.shape[1] == 1:
            pred_list = np.concatenate(pred_list, axis=0).astype(np.int)
            label_list = [np.array(e) for e in label_list]
            label_list = np.concatenate(np.array(label_list), axis=0)
        # print(pred_list)
        # print(label_list)
        print('accuracy: {} recall: {} precision: {} f1 score: {}'.format(accuracy_score(label_list, pred_list),
                                                                          recall_score(label_list, pred_list),
                                                                          precision_score(label_list, pred_list),
                                                                          f1_score(label_list, pred_list)))
