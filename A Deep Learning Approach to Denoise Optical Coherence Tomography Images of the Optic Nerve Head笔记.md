## 复现+笔记《A Deep Learning Approach to Denoise Optical Coherence Tomography Images of the Optic Nerve Head》

### 目前能想到的新的创新点，U-NET 的架构和 扩散模型具有天生的结合优势。利用扩散模型的多步生成特性，通过UNet的编码器解码器结构提取多尺度特征，将这些特征融合在扩散模型的生成过程中。这样理论会有更好的去噪效果。

### 代码复现 （pytorch）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(StandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.elu2 = nn.ELU()

    def forward(self, x):
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        identity = x
        out = self.elu1(self.bn1(self.conv1(x)))
        out = self.elu2(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        return out + identity

class OCTDenoisingNet(nn.Module):
    def __init__(self):
        super(OCTDenoisingNet, self).__init__()
        self.init_channels = 64

        # Downsampling Tower
        self.down1 = StandardBlock(self.init_channels, dilation=1)
        self.conv_down1 = nn.Conv2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1)

        self.down2 = ResidualBlock(self.init_channels, dilation=2)
        self.conv_down2 = nn.Conv2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1)

        self.down3 = ResidualBlock(self.init_channels, dilation=4)
        self.conv_down3 = nn.Conv2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1)

        # Bottleneck (latent space)
        self.bottleneck = StandardBlock(self.init_channels, dilation=1)

        # Upsampling Tower
        self.up1 = ResidualBlock(self.init_channels, dilation=4)
        self.tconv1 = nn.ConvTranspose2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up2 = ResidualBlock(self.init_channels, dilation=4)
        self.tconv2 = nn.ConvTranspose2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up3 = StandardBlock(self.init_channels, dilation=1)
        self.tconv3 = nn.ConvTranspose2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Multi-scale fusion layers
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.init_channels, self.init_channels, kernel_size=1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
            ) for _ in range(4)
        ])

        # Final Output
        self.final_conv = nn.Conv2d(self.init_channels * 5, 1, kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []

        # Downsampling
        x1 = self.down1(x)
        skips.append(x1)
        x = self.conv_down1(x1)

        x2 = self.down2(x)
        skips.append(x2)
        x = self.conv_down2(x2)

        x3 = self.down3(x)
        skips.append(x3)
        x = self.conv_down3(x3)

        # Bottleneck
        x4 = self.bottleneck(x)
        skips.append(x4)

        # Upsampling
        x = self.up1(x4)
        x = self.tconv1(x)
        x = x + F.interpolate(skips[2], size=x.shape[2:], mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = self.tconv2(x)
        x = x + F.interpolate(skips[1], size=x.shape[2:], mode='bilinear', align_corners=False)

        x = self.up3(x)
        x = self.tconv3(x)
        x = x + F.interpolate(skips[0], size=x.shape[2:], mode='bilinear', align_corners=False)

        # Multi-scale fusion
        fused = [F.interpolate(self.fuse_convs[i](skips[i]), size=x.shape[2:], mode='bilinear', align_corners=False) for i in range(4)]
        x = torch.cat([x] + fused, dim=1)

        # Output layer
        x = self.final_conv(x)
        x = self.tanh(x)
        return x
```
### 调用和输出
```python
model = OCTDenoisingNet()
model.eval()
dummy_input = torch.randn(1, 64, 496, 384)  # Simulated input with 64 feature channels
output = model(dummy_input)
```
```python
tensor([[[[-0.0765, -0.0318, -0.0448,  ..., -0.0875, -0.0102, -0.0804],
          [-0.0575, -0.0031, -0.1226,  ..., -0.0551,  0.0615,  0.0162],
          [ 0.0191,  0.0627,  0.0969,  ..., -0.0339,  0.0275, -0.0084],
          ...,
          [ 0.0656, -0.1134, -0.0059,  ...,  0.0018, -0.1280, -0.0882],
          [-0.1242,  0.0196, -0.0886,  ..., -0.1279, -0.0266,  0.0512],
          [-0.0792, -0.0649, -0.1312,  ...,  0.1052, -0.0417, -0.0307]]]],
       grad_fn=<TanhBackward0>)
```

## 笔记
### 一、研究背景与研究动因
光学相干断层扫描（Optical Coherence Tomography, OCT）是一种高度精细的非侵入式成像手段，能够实现对视网膜和视神经头等结构的亚细胞级别分辨率成像。由于其快速、无接触的特性，OCT 已经成为眼科诊疗中不可替代的成像标准，尤其在青光眼、视神经萎缩、年龄相关性黄斑变性等视神经疾病的筛查、诊断和进展追踪中发挥着重要作用。然而，在实际成像过程中，OCT 图像常受到 speckle 噪声的干扰，该类噪声具有强烈的空间相关性与非高斯分布特性，会严重影响图像对比度、组织边界识别能力和自动分割算法的鲁棒性。

虽然多帧信号平均（multi-frame averaging）方法可在一定程度上抑制散斑噪声，提升信噪比与组织结构清晰度，但这类方法依赖于长时间扫描与眼动跟踪稳定性，容易受到患者配合程度、微动与眼球漂移等因素影响，因此在临床应用中面临效率与舒适度的双重挑战。为此，本研究提出了一种基于深度卷积神经网络的端到端图像去噪框架，旨在利用单帧低质量图像生成具有多帧信号平均图像质量的高质量 OCT B 扫图像，从而在提升图像解读质量的同时显著降低扫描时长与患者负担。

---

### 二、网络架构与方法设计
该研究所设计的网络框架融合了多种经典与前沿的深度学习模块，在提升建模能力的同时，兼顾计算效率与结构重建的准确性。整体架构为 U-Net 类型的 encoder-decoder 框架，包含下采样路径、上采样路径、跳跃连接机制以及多尺度融合模块，关键结构特性如下：

- **空洞卷积（Dilated Convolutions）**：在保持感受野增长的同时不牺牲分辨率，使网络能够捕获长程上下文结构特征；
- **残差连接（Residual Learning）**：借鉴 ResNet 架构理念，引入恒等映射以缓解深层网络训练过程中的梯度消失与退化问题；
- **多尺度特征融合（Multi-scale Hierarchical Fusion）**：通过整合不同深度层次的特征图，有效增强组织边界和纹理细节的恢复能力；
- **跳跃连接（Skip Connections）**：将 encoder 中间层的高分辨率特征直接传递至 decoder 对应层，有助于低层细节的结构还原；
- **端到端训练**：输入输出保持相同维度（496×384），具备快速部署与推理能力，适应临床实时处理需求。

训练阶段，作者采用添加高斯噪声（均值 0，方差 1）模拟散斑效应，以 clean（多帧平均）图像与对应 noisy 图像配对作为监督信号，借助 L1 损失进行网络参数学习。

---

### 三、数据集构建与数据增强策略
研究共纳入 20 名健康志愿者，全部由新加坡国家眼科中心招募，每位受试者双眼均接受 OCT 成像。使用 Spectralis 系统采集单帧图像与多帧平均图像（每张 B 扫由 75 次平均获得），扫描区域覆盖视神经头中心 15° × 10° 区域，总体图像参数为 97 张 B-scan，每张含 384 个 A-scan。最终数据统计如下：

- **训练集**：来自 12 位受试者，共 2,328 对 clean/noisy 图像对；
- **测试集**：剩余 8 位受试者，共 1,552 张单帧图像；
- **数据增强**：离线执行，扩增至 23,280 个图像样本对，提升模型泛化能力。

#### 增强策略详解：
- **弹性形变（Elastic Deformation）**：模拟因青光眼等病变引起的解剖畸形；
- **旋转扰动（±10°）**：增强网络对成像角度变化的鲁棒性；
- **遮挡模拟（Occlusion Patches）**：模拟血管阴影或反射干扰，通过遮挡块降低局部可见度；
- **水平翻转**：扩大数据多样性，避免方向偏倚。

---

### 四、去噪性能评估：定量指标与实验结果
为全面量化网络的去噪性能，研究分别在整体图像层面与组织区域层面引入了多项评估指标，并在独立测试集上进行验证。

#### 核心指标定义：
- **信噪比（SNR）**：以 dB 为单位评估图像强度与噪声之间的比例；
- **对比噪声比（CNR）**：用于测量不同组织层之间的对比清晰度；
- **平均结构相似性指数（MSSIM）**：从结构、亮度与对比多个维度度量结构重建质量。

| 图像质量指标 | 单帧图像       | 去噪后图像     | 提升幅度 |
|----------------|----------------|----------------|------------|
| SNR (dB)       | 4.02 ± 0.68    | 8.14 ± 1.03    | ≈ 2 倍     |
| CNR            | 3.50 ± 0.56    | 7.63 ± 1.81    | ≈ 2.2 倍   |
| MSSIM          | 0.13 ± 0.02    | 0.65 ± 0.03    | ≈ 5 倍     |

#### 临床相关参数一致性分析：
在 8 名测试对象中，作者还评估了三项关键结构参数的测量误差：
- p-RNFLT（视神经纤维层厚度）：3.07% 平均误差；
- p-GCCT（神经节细胞+内丛状层厚度）：2.95%；
- p-CT（脉络膜厚度）：3.90%；

上述结果显示去噪图像在结构测量中具备高度一致性，具备临床可采信度。

---

### 五、图像可视化分析与主观评估
图像的主观质量评估由两位经验丰富的视网膜专家完成，评估指标包括结构完整性、边界清晰度与是否存在伪影。

- 去噪图像在所有案例中均未观察到由深度网络引入的人工伪影；
- ONH 区域内的 LC、RPE、IPL 等低强度区域表现出显著增强；
- 与多帧图像相比，去噪图在结构可视性方面表现接近，部分图像甚至优于后者；
- 未使用数据增强训练的网络虽然 SNR 提升更明显，但图像存在过度平滑、纹理缺失问题，降低了解剖学可读性。

---

### 六、方法优势总结
- **实时处理能力**：每张图像推理时间 < 20ms，适用于临床现场集成；
- **设备无关性高**：输出分辨率恒定，适配不同 OCT 图像输入尺寸；
- **低成本部署**：可作为嵌入式软件模块运行于现有 OCT 设备，无需额外硬件；
- **增强后处理系统性能**：为自动分割与结构厚度测量等任务提供更高质量输入；
- **临床友好性强**：从 3.5 分钟扫描时间缩短至 27 秒，提升患者体验。

---

### 七、局限性分析与技术挑战
1. **设备特异性训练**：当前网络仅基于 Spectralis 数据训练，跨设备泛化性尚需验证；
2. **散斑噪声建模不精确**：未使用真实 OCT 系统中的 speckle 分布模型，仅采用高斯噪声模拟；
3. **尚未验证病理图像性能**：全部训练与测试数据来自健康个体，缺乏病灶区域泛化能力评估；
4. **图像纹理过度平滑**：部分去噪图缺乏真实纹理信息，可能影响微观结构解读与定量分析；
5. **组织学配准缺失**：未引入组织切片对照，无法进行解剖层面的真实性校验；
6. **测试样本规模有限**：样本总量与人群多样性仍需扩展，以增强结论外推力。

---

### 八、未来研究方向与潜在拓展
- 开发更加符合 OCT 系统成像物理机制的噪声仿真器（如 Rayleigh 或 Gamma 分布）；
- 构建大规模多中心病理 OCT 数据集以评估模型在实际临床场景中的泛化表现；
- 结合无监督或半监督学习以减少对 clean 图像的依赖，拓展数据利用率；
- 与 OCT 血流成像、偏振成像等多模态系统融合，实现更完整的组织状态评估；
- 引入 Transformer 或注意力机制进一步提升远程特征建模与细节恢复能力。

---

### 结语与总结
本研究提出的深度学习去噪方法为 OCT 单帧图像质量增强提供了全新思路，通过端到端神经网络建模，在不依赖传统成像硬件改造的前提下实现了图像信噪比与结构可视性的双重提升。实验表明，该方法在维持图像结构完整性的同时显著降低了扫描时长，具有良好的临床实用性与扩展潜力。未来，随着 OCT 数据标准化与跨设备适配研究的深入，该方法有望成为智能成像系统中不可或缺的基础模块。

