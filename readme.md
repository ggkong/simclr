# 使用SimCLR和EfficientNet进行视网膜图像分类的实验总结

最近一直在看自监督的模型，想自己做一个试试自监督怎么搞，顺便熟悉下任务。

具体过程是通过SimCLR对比学习的框架，先在没有标签的图像上进行预训练，让模型学会提取有用的图像特征。

然后再在有标签的数据集上进行微调，用来做DR的分级预测。

主干网络是EfficientNet-B0，它在参数量不多的情况下也有不错的表现，适合这种中小型医学图像任务。

整个流程主要分为两个阶段：第一阶段是用SimCLR做自监督预训练，第二阶段是在APTOS 2019数据集上做分类微调。

然后画了几张图。所有的代码都上传的GitHub：https://github.com/ggkong/simclr.git

**NOTE: 因为目前我只拿到了APTOS 2019的数据集，所以预训练过程也是使用了APTOS数据集 而且预训练只用了1000张图片。**

1、aptos_dataset.py 数据加载 + 图像增强

没有选用torchvision.transforms，而是用了更多功能的albumentations做增强

图像用224，增强用了: 随机裁剪一个区域,随机水平翻转,随机调整亮度、对比度、饱和度、色调。

最终的返回是：得到两个不同的视图 `xi` 和 `xj`（正样本对）

用于 SimCLR 的正样本对训练

```PYTHON
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class APTOSSimCLRDataset(Dataset):
    def __init__(self, image_dir, image_size=224):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        xi = self.transform(image=image)['image']
        xj = self.transform(image=image)['image']
        return xi, xj
```

2、simclr_model.py  EfficientNet + MLP

**Tip：这里我使用了本地下载后再加载，因为链接huggingface的时候老是出错，所以直接用了：**

```python
import timm
print(timm.models.create_model('efficientnet_b0').default_cfg)
```

这两行代码读取下载路径 然后下载到本地：
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth', 'hf_hub_id': 'timm/efficientnet_b0.ra_in1k', 'architecture': 'efficientnet_b0', 'tag': 'ra_in1k', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': False, 'interpolation': 'bicubic', 'crop_pct': 0.875, 'crop_mode': 'center', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 1000, 'pool_size': (7, 7), 'first_conv': 'conv_stem', 'classifier': 'classifier'}

这就是下载路径。

**NOTE：SimCLR 论文中提到要加一个投影头，把 `h`（编码器输出）映射到 `z`（对比空间），MLP充当映射器。**


```python
import torch.nn as nn
import timm
import torch


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', projection_dim=128,
                 local_weight_path='efficientnet_b0.pth'):
        super().__init__()

        # 创建 EfficientNet 模型
        self.encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        # 本地文件加载
        state_dict = torch.load(local_weight_path, map_location='cpu')
        self.encoder.load_state_dict(state_dict, strict=False)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
```

3、contrastive_loss.py SimCLR 对比损失 NT-Xent

**用来判断 z1 和 z2（同一张图的两个增强视图）是否彼此靠近，同时远离其他图像的表示**
**让正样本（xi,xj）相似度最大，同时让它和其他样本（负样本）拉开距离，也就是很多论文中提到的。**

```python
import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2N
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)

    mask = torch.eye(2*N, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2*N, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
    labels = torch.zeros(2*N, dtype=torch.long).to(z.device)

    return F.cross_entropy(logits / temperature, labels)
```

4、train

```py
import torch
from torch.utils.data import DataLoader
from aptos_dataset import APTOSSimCLRDataset
from simclr_model import SimCLRModel
from contrastive_loss import nt_xent_loss
import os
import torch
torch.cuda.empty_cache()

def main():
    device = 'cuda'
    dataset = APTOSSimCLRDataset(image_dir='train_images')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 50
    save_interval = 1  

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for xi, xj in loader:
            xi, xj = xi.to(device), xj.to(device)

            zi = model(xi)
            zj = model(xj)
            loss = nt_xent_loss(zi, zj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {epoch_loss:.4f}")

        if epoch % save_interval == 0 or epoch == num_epochs:
            save_path = f'checkpoints/simclr_epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
```

**NOTE：没有训到50epochs 就被停下了，因为老是爆显存，而且我看loss下降的太快了，而且很低。**

**原因推测：**

**1、数据量太小了，所以太好训练了。**

**2、Loss 可以修改一下，这个loss的作用性不是很强。**

```python
Epoch [1/50] Loss: 1.2034
模型已保存至 checkpoints/simclr_epoch_1.pth
Epoch [2/50] Loss: 0.0001
模型已保存至 checkpoints/simclr_epoch_2.pth
Epoch [3/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_3.pth
Epoch [4/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_4.pth
Epoch [5/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_5.pth
Epoch [6/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_6.pth
Epoch [7/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_7.pth
Epoch [8/50] Loss: 0.0021
模型已保存至 checkpoints/simclr_epoch_8.pth
Epoch [9/50] Loss: 0.0000
模型已保存至 checkpoints/simclr_epoch_9.pth
```

![image-20250325153229623](C:\Users\kongge\AppData\Roaming\Typora\typora-user-images\image-20250325153229623.png)

14G显存占用量。

5、开始微调做下游任务。我的数据源是相同的，就类似于论文中的内部数据集。

```python
# 构建内部数据集
class APTOSClassifyDataset(Dataset):
    def __init__(self, image_dir, csv_file, image_size=224, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mode = mode
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5 if mode == 'train' else 0.0),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['id_code'] + '.png')
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        label = row['diagnosis']
        return image, label
```

```python
# 加载模型 加上分类的MLP
class EfficientNetClassifier(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes=5, pretrained_encoder_path=None):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        
        if pretrained_encoder_path:
            state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=False)

        self.classifier = nn.Linear(self.encoder.num_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
```

```python
device = torch.device('cuda' )
train_dataset = APTOSClassifyDataset('fine_tune/train_split', 'fine_tune/train.csv', mode='train')
test_dataset = APTOSClassifyDataset('fine_tune/test_split', 'fine_tune/test.csv', mode='test')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
model = EfficientNetClassifier(pretrained_encoder_path='checkpoints/simclr_epoch_9.pth').to(device)
```

```python
# 交叉熵 + Adam
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

**NOTE：一开始训的时候发现loss 降得慢而且val 也很差 基本没用，分析数据，是样本不平衡问题**

```
Epoch 1/20 - Train Loss: 1.5337 - Val Acc: 0.3000
Epoch 2/20 - Train Loss: 1.4284 - Val Acc: 0.3000
Epoch 3/20 - Train Loss: 1.3271 - Val Acc: 0.3000
Epoch 4/20 - Train Loss: 1.1888 - Val Acc: 0.3000
Epoch 5/20 - Train Loss: 1.0532 - Val Acc: 0.3000 
```

```python
df = pd.read_csv('fine_tune/train.csv')
print(df['diagnosis'].value_counts())
```

```python
diagnosis
0    346
2    186
1     71
4     58
3     38
Name: count, dtype: int64
```

就是因为把所有的都预测成了0所以ACC才一直是0.3

NOTE：修改weight 稍微解决一下这个问题，其实还有其他的解决方案 例如数据合成。

```python
train_df = pd.read_csv('fine_tune/train.csv')
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_df['diagnosis']),
                                     y=train_df['diagnosis'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
```

```Python
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    # === 验证 ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch}/{num_epochs} "
          f"- Train Loss: {train_loss / len(train_loader):.4f} "
          f"- Train Acc: {train_acc:.4f} "
          f"- Val Acc: {val_acc:.4f}")
```

```python
Epoch 1/20 - Train Loss: 1.5545 - Train Acc: 0.3763 - Val Acc: 0.3000
Epoch 2/20 - Train Loss: 1.4562 - Train Acc: 0.4764 - Val Acc: 0.3000
Epoch 3/20 - Train Loss: 1.3632 - Train Acc: 0.5622 - Val Acc: 0.3000
Epoch 4/20 - Train Loss: 1.2550 - Train Acc: 0.5880 - Val Acc: 0.3000
Epoch 5/20 - Train Loss: 1.1213 - Train Acc: 0.6252 - Val Acc: 0.3000
Epoch 6/20 - Train Loss: 1.0129 - Train Acc: 0.6552 - Val Acc: 0.3300
Epoch 7/20 - Train Loss: 0.8905 - Train Acc: 0.6924 - Val Acc: 0.5767
Epoch 8/20 - Train Loss: 0.7871 - Train Acc: 0.7096 - Val Acc: 0.6933
Epoch 9/20 - Train Loss: 0.7322 - Train Acc: 0.7182 - Val Acc: 0.6933
Epoch 10/20 - Train Loss: 0.7150 - Train Acc: 0.7239 - Val Acc: 0.7100
Epoch 11/20 - Train Loss: 0.6651 - Train Acc: 0.7396 - Val Acc: 0.7000
Epoch 12/20 - Train Loss: 0.6277 - Train Acc: 0.7425 - Val Acc: 0.7000
```

这里效果就好了一些，当然也有可能是训练不到位决定的，我怕爆显存，训到12epoch结束。val acc0.7，效果还行，

**NOTE：但是有一个问题没有验证：是MLP决定了效果变好还是预训练模型决定了效果变好呢? 需要加上可解释性才能决定，或者设计其他的实验。**

```python
os.makedirs('finetune_checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'finetune_checkpoints/finetuned_model.pth')
```

剩下的就是各种图：

![image-20250325154515006](C:\Users\kongge\AppData\Roaming\Typora\typora-user-images\image-20250325154515006.png)

![image-20250325154521122](C:\Users\kongge\AppData\Roaming\Typora\typora-user-images\image-20250325154521122.png)

![image-20250325154529660](C:\Users\kongge\AppData\Roaming\Typora\typora-user-images\image-20250325154529660.png)

![image-20250325154538247](C:\Users\kongge\AppData\Roaming\Typora\typora-user-images\image-20250325154538247.png)
