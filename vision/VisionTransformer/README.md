# Vision Transformer (ViT) from Scratch - Pretraining on TinyImageNet, Finetuning on CIFAR-10

## 📌 Overview

This project implements a Vision Transformer (ViT) from scratch in PyTorch.
It includes pretraining on the **TinyImageNet** dataset and fine-tuning on **CIFAR-10**.

---

## 📁 Project Structure

```
.
├── model.py          # ViT model implementation
├── patchdata.py      # Patch generator + custom dataset handler
├── vit.py            # Train / Finetune entry point
├── test.py           # Evaluation script
├── logs/             # TensorBoard logs (if enabled)
└── README.md         # This file
```

---

## 🧐 Model Architecture

* **Backbone:** Vision Transformer Tiny (ViT-Ti)
* **Input size:** 224x224
* **Patch size:** 16x16
* **Number of layers:** 12
* **Hidden dimension (latent vector):** 128
* **MLP dimension:** 512
* **Heads:** 3
* **Dropout:** 0.1

---

## 🔧 Training Configuration

### Pretraining (TinyImageNet)

| Hyperparameter | Value        |
| -------------- | ------------ |
| Dataset        | TinyImageNet |
| Image size     | 224          |
| Patch size     | 16           |
| Epochs         | 300          |
| Batch size     | 128          |
| Learning rate  | 5e-4         |
| Weight decay   | 0.05         |
| Dropout        | 0.1          |
| Num classes    | 200          |

### Fine-tuning (CIFAR-10)

| Hyperparameter | Value    |
| -------------- | -------- |
| Dataset        | CIFAR-10 |
| Image size     | 224      |
| Patch size     | 16       |
| Epochs         | 100      |
| Batch size     | 64       |
| Learning rate  | 0.0001   |
| Weight decay   | 0.0001   |
| Dropout        | 0.1      |
| Num classes    | 10       |

---

## 📊 Results

| Stage       | Dataset      | Test Accuracy | Notes               |
| ----------- | ------------ | ------------- | ------------------- |
| Pretraining | TinyImageNet | **00.00**     |                     |
| Fine-tuning | CIFAR-10     | **77.88%**    |                     |

> *Note: Accuracy may vary based on GPU and hyperparameter tuning.*

---

## 📊 TensorBoard (Optional)

If you want to track your training with TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/exp1')
writer.add_scalar("Loss/train", loss.item(), epoch)
writer.add_scalar("Acc/train", acc, epoch)
```

Launch with:

```bash
tensorboard --logdir=runs
```

---

## 📌 How to Run

### Pretrain on TinyImageNet

```bash
python vit.py --mode train --dataname tinyimagenet --epochs 300 --img_size 224 --num_classes 200 --dataname tiny-imagenet
```

### Finetune on CIFAR-10

```bash
python vit.py --mode train --dataname cifar10 --epochs 100 --img_size 224 --num_classes 10 --pretrained 1 --pretrained_path ./model.pth
```

### Test

```bash
python test.py --mode test --dataname cifar10 --pretrained_path ./model.pth
```

---

## 📀 Dependencies

```bash
torch==2.0.1
torchvision
tensorboard
numpy
Pillow
```

---

## 🙌 Acknowledgements

* [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
* CIFAR-10 dataset via `torchvision.datasets`
* TinyImageNet dataset from Stanford CS231n

---

## 💡 Future Work

* Support more ViT variants (ViT-B/16, ViT-L/32)
* Add positional embedding visualization
* Extend to object detection tasks
