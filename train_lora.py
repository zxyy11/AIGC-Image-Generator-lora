import os
# --- 破壁魔法开始（解决 SSL 拦截与加速下载） ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# --- 破壁魔法结束 ---

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ================= 1. 炼丹超参数配置 =================
# 核心修改：把基础大模型换成了专精二次元和风景的 Counterfeit 神模！
MODEL_ID = "gsdf/Counterfeit-V2.5"          

DATA_DIR = "./dataset/image/50_madias"                # 数据集路径
OUTPUT_DIR = "./output_lora"                # 练好的模型保存路径
RANK = 32                                   # LoRA 的脑容量 (Rank)
LEARNING_RATE = 5e-5                        # 学习率
EPOCHS = 3                                 # 训练轮数 (完美卡在黄金甜点区)
BATCH_SIZE = 2                              # 3090 显存够大，可以一次吃 2 张图
RESOLUTION = 512                            # 训练分辨率

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. 数据集加载器 (兼容 100_ 文件夹格式) =================
class MadiasDataset(Dataset):
    def __init__(self, folder, tokenizer, size):
        self.image_paths = []
        self.captions = []
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # 归一化到 [-1, 1]
        ])

        print("🔍 正在解析文件夹格式...")
        for root, dirs, files in os.walk(folder):
            folder_name = os.path.basename(root)
            repeats = 1
            # 自动识别 100_madias 这种格式并设置循环次数
            if '_' in folder_name and folder_name.split('_')[0].isdigit():
                repeats = int(folder_name.split('_')[0])

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(root, file)
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        self.image_paths.extend([img_path] * repeats)
                        self.captions.extend([caption] * repeats)

        print(f"📦 发现 {len(set(self.image_paths))} 张独立图片。")
        print(f"🔄 按照循环倍数计算，每个 Epoch 投喂数据量为: {len(self.image_paths)} 样本。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)

        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids.squeeze()}

# ================= 3. 核心炼丹炉组装 =================
print("⏳ 正在把神级底模搬运到 3090 显存中...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# 冻结基础模型（不破坏它们原本的记忆框架）
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# 动态外挂：给 UNet 植入 PEFT LoRA 记忆芯片
print("🔌 正在给 UNet 外接 LoRA 记忆芯片...")
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 锁定核心的注意力机制网络层
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters() 

# ================= 4. 点火开炼 =================
dataset = MadiasDataset(DATA_DIR, tokenizer, RESOLUTION)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler() # 启用混合精度，极致优化 3090 显存

print("🚀 燃料加注完毕！3090 引擎全开，开始炼丹！")
unet.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # 1. 编码器压缩图像
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # 2. 制造随机噪声并融合
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. 提取提示词语义特征
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        # 4. 混合精度前向传播 (预测噪声)
        with torch.autocast("cuda"):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # 5. 反向传播与梯度更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    print(f"✅ 第 {epoch+1} 轮结束, 平均 Loss: {total_loss/len(dataloader):.4f}")

# ================= 5. 保存胜利果实 =================
save_path = os.path.join(OUTPUT_DIR, "madias_style_lora")
unet.save_pretrained(save_path)
print(f"🎉 炼丹大功告成！专属 LoRA 权重已保存在: {save_path}")