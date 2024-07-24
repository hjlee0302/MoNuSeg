import torch

def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler,
                    print_freq=10, min_valid_loss=100):
    for train_iter, pack in enumerate(train_data_loader):
        train_loss = 0
        valid_loss = 0
        img = pack['img'].to(device)
        label = pack['label'].to(device)
        optimizer.zero_grad()
        pred = model(img)
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
                else:
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}' \
                          .format(epoch + 1, train_iter + 1, len(train_data_loader), train_loss / print_freq,
                                  valid_loss / len(valid_data_loader), lr_scheduler.get_last_lr()))

        lr_scheduler.step()
    return min_valid_loss