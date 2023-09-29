from collections import Counter
import random
import os
import cv2
import matplotlib.pyplot as plt




def check_file_type(image_folder):
    extension_type = []
    file_list = os.listdir(image_folder)
    for file in file_list:
        extension_type.append(file.rsplit(".", 1)[1].lower())
    print(Counter(extension_type).keys())
    print(Counter(extension_type).values())
    
    
def check_image_size(image_folder):
    total_img_list = os.listdir(image_folder)
    counter = 0
    for image in total_img_list:
        try:
            img_path = os.path.join(f'{image_folder}/{image}')
            img = cv2.imread(img_path)
            print(img.shape)
        except:
            print("This {} is problematic.".format(image))
    return counter 


def display_img(dir,  n=1):

    img_names = os.listdir(dir)
    random.shuffle(img_names)
    img_names = img_names[:n]
    for name in img_names:
        print(name)
        img_path = os.path.join(f'{dir}/{name}')
        img = cv2.imread(img_path)
        print(img.shape)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

    plt.show()


def plot_metrics(losses, scores):
    train_losses = [loss[0] for loss in losses]
    val_losses = [loss[1] for loss in losses]

    dice = [score[0] for score in scores]
    iou = [score[1] for score in scores]
    
    epochs = len(losses)
    plt.figure(figsize=(12, 5))

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    # Plot Dice and IoU scores
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), dice, label='Dice Score')
    plt.plot(range(1, epochs + 1), iou, label='IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Dice and IoU Scores')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
