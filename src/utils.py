import torch
import albumentations as A

class DiceCoef(torch.nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        y_pred = torch.round((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))
        intersection = (y_true * y_pred).sum()
        dice = (2.0 * intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
        
        return dice


class IoU(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true, threshold=0.5, smooth=1):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        y_pred = (y_pred > threshold).float()
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)

        return iou


class MultiBCELossFusion:
    def __init__(self):
        self.loss_function = torch.nn.BCELoss()

    def bce_loss(self, d, labels_v):
        loss = self.loss_function(d, labels_v)
        return loss

    def calculate_loss(self, d_list, labels_v):
        loss_list = [self.bce_loss(d, labels_v) for d in d_list]
        loss0 = loss_list[0]
        loss = loss0 * 1.5 + sum(loss_list[1:])
        return loss0, loss
    

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.9),
        A.Transpose(p=0.6),
        #A.Perspective()
    ], p=1.)

def get_valid_transform():
    return A.Compose([])