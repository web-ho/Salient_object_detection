class config:

    seed = 29
    epochs = 2
    batch_sz = 2

    img_sz = 512, 512


    train_img_dir = 'dataset/train/Image/'
    train_mask_dir = 'dataset/train/Mask/'

    valid_img_dir = 'dataset/valid/image/'
    valid_mask_dir = 'dataset/valid/mask/'

    pretrained_model_path = 'u2net.pth'
