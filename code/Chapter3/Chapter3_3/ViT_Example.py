import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
import cv2
import matplotlib
from matplotlib import font_manager


# 设置中文字体
def set_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试使用系统中已有的中文字体
        possible_fonts = [
            'SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong',
            'Arial Unicode MS', 'DejaVu Sans'  # 后备字体
        ]

        for font_name in possible_fonts:
            if font_name in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体: {font_name}")
                return True

        print("警告: 未找到合适的中文字体，可能显示乱码")
        return False
    except:
        print("字体设置失败，使用默认字体")
        return False


# 设置中文字体
set_chinese_font()


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """简化的多头自注意力机制"""

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # 图像分块与投影
        x = self.patch_embed(x)

        # 添加[class] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # 通过Transformer块
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits


def load_and_preprocess_image(image_path_or_url, img_size=224):
    """加载和预处理图像"""
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor


def visualize_patches(image, patch_size=16):
    """可视化图像分块过程"""
    img_array = np.array(image)
    h, w, c = img_array.shape
    grid_h, grid_w = h // patch_size, w // patch_size

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 原始图像
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 显示分块网格
    axes[1].imshow(img_array)
    for i in range(0, h, patch_size):
        axes[1].axhline(i, color='red', alpha=0.3)
    for j in range(0, w, patch_size):
        axes[1].axvline(j, color='red', alpha=0.3)
    axes[1].set_title(
        f'Image Patches ({patch_size}x{patch_size} pixels/patch)\nTotal: {grid_h}x{grid_w}={grid_h * grid_w} patches',
        fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # 显示一些示例块
    print(f"\nExample Image Patches (first 4):")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for idx in range(4):
        i, j = idx // grid_w, idx % grid_w
        patch = img_array[i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size]
        axes[idx].imshow(patch)
        axes[idx].set_title(f'Patch {idx + 1}\nPosition({i},{j})')
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def demonstrate_vit_on_real_image(image_path, model):
    """在真实图像上演示ViT工作原理"""
    print("=" * 60)
    print("ViT Working Principle Demonstration")
    print("=" * 60)

    # 加载图像
    original_image, processed_tensor = load_and_preprocess_image(image_path)

    print(f"Original image size: {original_image.size}")
    print(f"Processed tensor shape: {processed_tensor.shape}")

    # 可视化分块过程
    visualize_patches(original_image)

    # 模拟ViT前向传播的各个阶段
    with torch.no_grad():
        # 1. 图像分块与投影
        patch_embed = model.patch_embed
        patch_embeddings = patch_embed(processed_tensor)
        print(f"\n1. After patch projection: {patch_embeddings.shape}")
        print(f"   - Batch size: 1")
        print(f"   - Number of patches: {patch_embeddings.shape[1]}")
        print(f"   - Embedding dimension: {patch_embeddings.shape[2]}")

        # 2. 添加[class] token
        cls_tokens = model.cls_token.expand(1, -1, -1)
        sequence_with_cls = torch.cat((cls_tokens, patch_embeddings), dim=1)
        print(f"\n2. After adding [class] token: {sequence_with_cls.shape}")
        print(f"   - Sequence length: {sequence_with_cls.shape[1]} (196 image patches + 1 [class] token)")

        # 3. 添加位置编码
        sequence_with_pos = sequence_with_cls + model.pos_embed
        print(f"\n3. After positional encoding: {sequence_with_pos.shape}")

        # 4. 完整模型前向传播
        output = model(processed_tensor)
        print(f"\n4. Final output: {output.shape}")
        print(f"   - Number of class scores: {output.shape[1]}")

    return original_image, processed_tensor, output


def create_attention_visualization(image, model, processed_tensor):
    """创建注意力可视化"""
    print("\n" + "=" * 60)
    print("Attention Mechanism Visualization")
    print("=" * 60)

    img_array = np.array(image)
    h, w = img_array.shape[:2]
    patch_size = 16

    # 创建模拟的注意力热图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 分块网格
    axes[0, 1].imshow(img_array)
    for i in range(0, h, patch_size):
        axes[0, 1].axhline(i, color='white', alpha=0.5, linewidth=0.5)
    for j in range(0, w, patch_size):
        axes[0, 1].axvline(j, color='white', alpha=0.5, linewidth=0.5)
    axes[0, 1].set_title('Image Patch Grid', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 模拟注意力热图1 - 关注中心区域
    attention_map1 = np.zeros((h // patch_size, w // patch_size))
    center_i, center_j = h // (2 * patch_size), w // (2 * patch_size)
    for i in range(attention_map1.shape[0]):
        for j in range(attention_map1.shape[1]):
            dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
            attention_map1[i, j] = np.exp(-dist / 2)

    # 模拟注意力热图2 - 关注边缘区域
    attention_map2 = np.zeros((h // patch_size, w // patch_size))
    for i in range(attention_map2.shape[0]):
        for j in range(attention_map2.shape[1]):
            if i == 0 or i == attention_map2.shape[0] - 1 or j == 0 or j == attention_map2.shape[1] - 1:
                attention_map2[i, j] = 1.0

    # 调整热图大小以匹配原图
    attention_map1_resized = cv2.resize(attention_map1, (w, h), interpolation=cv2.INTER_NEAREST)
    attention_map2_resized = cv2.resize(attention_map2, (w, h), interpolation=cv2.INTER_NEAREST)

    # 显示注意力热图
    axes[1, 0].imshow(img_array)
    axes[1, 0].imshow(attention_map1_resized, alpha=0.6, cmap='hot')
    axes[1, 0].set_title('Attention Pattern 1: Focus on Center', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_array)
    axes[1, 1].imshow(attention_map2_resized, alpha=0.6, cmap='hot')
    axes[1, 1].set_title('Attention Pattern 2: Focus on Edges', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nAttention Mechanism Explanation:")
    print("• ViT uses self-attention to let each patch interact with all other patches")
    print("• The [class] token learns to focus on the most relevant image regions")
    print("• Different attention heads may focus on different aspects of the image")


def main():
    # 初始化ViT模型
    vit_model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=384,
        depth=6,
        num_heads=8
    )

    # 使用示例图像
    image_urls = [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",  # dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # cat
    ]

    for i, image_url in enumerate(image_urls[:1]):
        print(f"\n{'#' * 80}")
        print(f"Example {i + 1}: Analyzing Image")
        print(f"{'#' * 80}")

        try:
            original_image, processed_tensor, output = demonstrate_vit_on_real_image(
                image_url, vit_model
            )

            create_attention_visualization(original_image, vit_model, processed_tensor)

            # 显示模型输出解释
            print(f"\nModel Output Analysis:")
            print(f"• Output shape: {output.shape} → 1000 ImageNet class scores")
            print(f"• Max score: {torch.max(output).item():.3f}")
            print(f"• Mean score: {torch.mean(output).item():.3f}")
            print(f"• Model learns to map image features to class space")

        except Exception as e:
            print(f"Error processing image: {e}")
            continue

    # 技术细节说明
    print(f"\n{'=' * 80}")
    print("ViT Technical Details Summary")
    print(f"{'=' * 80}")
    print("""
1. Image Patching:
   - Input: [1, 3, 224, 224] (batch, channels, height, width)
   - Patches: 224×224 → 196 patches of 16×16
   - Output: [1, 196, 384] (batch, num_patches, embed_dim)

2. Linear Projection:
   - Each 16×16×3=768 pixel patch → 384-dim vector
   - Uses learnable projection matrix

3. [class] Token:
   - Learnable classification token
   - Prepended to sequence: [1, 197, 384] (196 patches + 1 [class] token)

4. Positional Encoding:
   - Adds learnable position vectors
   - Preserves spatial information

5. Transformer Encoding:
   - Multi-head self-attention: global information exchange
   - Feed-forward network: feature transformation
   - Layer normalization and residual connections

6. Classification Head:
   - Uses [class] token output
   - Linear layer maps to class scores
    """)


if __name__ == "__main__":
    main()