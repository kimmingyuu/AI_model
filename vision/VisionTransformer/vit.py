import patchdata
import model
import test
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--save_acc', default=50, type=int, help='val acc')
    parser.add_argument('--epochs', default=300, type=int, help='training epoch')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--latent_vec_dim', default=192, type=int, help='latent dimension')
    parser.add_argument('--num_heads', default=3, type=int, help='number of heads')
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer')
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name')
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')
    parser.add_argument('--pretrained_path', default='./model.pth', type=str, help='pretrained model path')
    args = parser.parse_args()
    print(args)

    latent_vec_dim = args.latent_vec_dim
    mlp_hidden_dim = int(latent_vec_dim/2)
    # mlp_hidden_dim = latent_vec_dim * 4 # for pre-training CIFAR-100
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Image Patches
    d = patchdata.Flattened2Dpatches(dataname=args.dataname, img_size=args.img_size, patch_size=args.patch_size,
                                     batch_size=args.batch_size)
    trainloader, valloader, testloader = d.patchdata()
    image_patches, _ = next(iter(trainloader))

    # Model
    vit = model.VisionTransformer(patch_vec_size=image_patches.size(2), num_patches=image_patches.size(1),
                                  latent_vec_dim=latent_vec_dim, num_heads=args.num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                  drop_rate=args.drop_rate, num_layers=args.num_layers, num_classes=args.num_classes).to(device)
    writer = SummaryWriter(log_dir='./runs/vit_tinyimagenet')

    if args.pretrained == 1:
        # Load pretrained weights
        # pretrained_dict = torch.load(args.pretrained_path)
        # pretrained_dict = torch.load(args.pretrained_path)
        # model_dict = vit.state_dict()
        
        # # Filter out mlp_head layer parameters
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'mlp_head' not in k}
        # model_dict.update(pretrained_dict)
        # vit.load_state_dict(model_dict, strict=False)
        vit.load_state_dict(torch.load(args.pretrained_path))
        # print("Loaded pretrained model weights (except classification head)")

    if args.mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.AdamW(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay) # for pre-training CIFAR-100
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
        
        #optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs)

        # Train
        n = len(trainloader)
        best_acc = args.save_acc
        for epoch in range(args.epochs):
            running_loss = 0
            for img, labels in tqdm(trainloader, desc=f'Epochs: {epoch+1}/{args.epochs}', position=0): 
                optimizer.zero_grad()
                outputs, _ = vit(img.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                scheduler.step()

            train_loss = running_loss / n
            val_acc, val_loss = test.accuracy(valloader, vit, (args.batch_size, num_patches, image_patches.size(2)))
            tqdm.write(f'[{epoch}] train loss: {train_loss:.3f}, validation loss: {val_loss:.3f}, validation acc {val_acc:.2f} %')
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                tqdm.write(f'[{epoch}] Saved best model - train loss: {train_loss:.3f}, validation loss: {val_loss:.3f}, validation acc: {val_acc:.2f}%')
                torch.save(vit.state_dict(), './model.pth')

    else:
        test_acc, test_loss = test.accuracy(testloader, vit, (args.batch_size, num_patches, image_patches.size(2)))
        print('test loss: %.3f, test acc %.2f %%' % (test_loss, test_acc))
