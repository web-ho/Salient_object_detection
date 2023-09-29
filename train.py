import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import create_dataframe, Mydataset
from src.engine import train_loop, validation_loop
from src.utils import get_train_transform
from src.cfg import config

from model import U2NETP, U2NET


# training script
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = create_dataframe(config.train_img_dir, config.train_mask_dir)
    valid_df = create_dataframe(config.valid_img_dir, config.valid_mask_dir)

    train_dataset = Mydataset(train_df, get_train_transform())
    valid_dataset = Mydataset(valid_df)

    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=0
                            )
    valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=0
                            )

    if args.pretrain:
        model = U2NET(in_ch=3, out_ch=1) 
        model.load_state_dict(torch.load(config.pretrained_model_path))
        num_epochs = config.pretrain_epochs
    else:
        model = U2NET(in_ch=3, out_ch=1) 
        num_epochs = config.epochs

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.001, 
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0
                                )

    losses = []
    scores = []
    for epoch in range(1, num_epochs+1):
        print(f'Starting epoch: {epoch}')

        train_loss = train_loop(model, 
                        train_loader, 
                        optimizer, 
                        device
                        )
        val_loss, dice, iou = validation_loop(model,
                                    valid_loader,
                                    device
                                    )
        
        losses.append((train_loss, val_loss))
        scores.append((dice, iou))
        print(f"Train_loss:{train_loss:.4f}, Valid_loss:{val_loss:.4f}\n, Dice_score:{dice:.4f}, Iou_score:{iou:.4f}")
        
    torch.save(model.state_dict(), f"model_epoch_{num_epochs}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true", help="Use a pretrained model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    args = parser.parse_args()
    main(args)
