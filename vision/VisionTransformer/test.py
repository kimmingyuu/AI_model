import torch
import torch.nn as nn
from torchinfo import summary

def print_model_summary(model, input_size):
    """
    Args:
        model
        input_size: (batch_size, num_patches, patch_vec_size)
        num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size))
        patch_vec_size = 3 * args.patch_size * args.patch_size  # 3 is for RGB channels
    """
    summary(model, input_size=input_size)

def accuracy(dataloader, model, input_size):
    correct = 0
    total = 0
    running_loss = 0
    n = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
            running_loss += loss.item()

        loss_result = running_loss / n
    # print_model_summary(model, input_size)
    acc = 100 * correct / total
    model.train()
    return acc, loss_result