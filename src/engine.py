from tqdm import tqdm
import torch

from src.utils import DiceCoef, IoU, MultiBCELossFusion, normPRED


def train_loop(model, dataloader, optimizer, device, scheduler=None):

    model.train()  
    model.to(device)
    optimizer.zero_grad()

    running_loss = 0.0

    for batch in tqdm(dataloader, total=len(dataloader)):
        image = batch[0]
        mask = batch[1]
        image = image.to(device)
        mask = mask.to(device)

        d0, d1, d2, d3, d4, d5, d6 = model(image)
        _, loss = MultiBCELossFusion().calculate_loss([d0, d1, d2, d3, d4, d5, d6], mask)

        loss.backward() 
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
    train_loss = running_loss / len(dataloader)

    return train_loss


def validation_loop(model, dataloader, device):

    dice_coef = DiceCoef()
    iou = IoU()

    dice_score = []
    iou_score = []
    total_dice_score = 0
    total_iou_score = 0

    running_loss = 0.0

    model.eval()
    model.to(device)

    for image, mask in tqdm(dataloader, total=len(dataloader)):
        image = image.to(device)
        mask = mask.to(device)

        d0, d1, d2, d3, d4, d5, d6 = model(image)
        pred = d0[:,0,:,:]
        pred = normPRED(pred)
        _, loss = MultiBCELossFusion().calculate_loss([d0, d1, d2, d3, d4, d5, d6], mask)

        score = dice_coef(pred, mask).item()
        scores = iou(pred, mask).item()
        dice_score.append(score)
        iou_score.append(scores)

        total_dice_score += score
        total_iou_score += scores
        
        running_loss += loss.item()
    # Calculate the average (mean) Dice and IoU scores
    val_loss = running_loss / len(dataloader)
    avg_dice = total_dice_score / len(dataloader)
    avg_iou = total_iou_score / len(dataloader)

    return val_loss, avg_dice, avg_iou
