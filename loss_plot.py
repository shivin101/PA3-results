from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import torch
import os
import sys

def plot_loss(loss_file,x_label='Iterations',y_label='BCE Loss',filename='loss.pdf',title='Loss Graph'):
    with open(loss_file, 'rb') as f:
        [train_losses,val_losses]=pkl.load(f)
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join('./', filename))
    plt.close()

file1 = sys.argv[1]
save_name = sys.argv[2]
plot_loss(file1,title='ResNet Fine Tuning',filename=save_name)
