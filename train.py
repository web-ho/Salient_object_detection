import torch
from torch.utils.data import DataLoader
from src.dataset import create_dataframe, Mydataset
from src.engine import train_loop, validation_loop
from src.utils import get_train_transform
from src.cfg import config

from model import U2NETP, U2NET

# training script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = create_dataframe(config.train_img_dir, config.train_mask_dir)
    valid_df = create_dataframe(config.valid_img_dir, config.valid_mask_dir)

    train_dataset = Mydataset(train_df, get_train_transform())
    valid_dataset = Mydataset(valid_df)

    train_loader = DataLoader(train_dataset,
                            batch_size=2, 
                            shuffle=True, 
                            num_workers=0
                            )
    valid_loader = DataLoader(valid_dataset,
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0
                            )

    model = U2NET(in_ch=3, out_ch=1) 
    model.load_state_dict(torch.load(config.pretrained_model_path))

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.001, 
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0
                                )

    losses = []
    scores = []
    for epoch in range(1, config.epochs+1):
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
        
    torch.save(model.state_dict(), f"model_epoch_{config.epochs}.pth")
