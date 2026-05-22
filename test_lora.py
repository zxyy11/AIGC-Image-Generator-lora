import torch
import os
# --- 解决 SSL 证书问题和加速下载 --- 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from diffusers import StableDiffusionPipeline

print("⏳ 正在组装基础大模型与专属 LoRA 记忆芯片...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用与训练时相同的二次元大模型 Counterfeit-V2.5
model_id = "gsdf/Counterfeit-V2.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.load_lora_weights("./output_lora/madias_style_lora")
pipe.to(device)

print("🚀 开始施展强效魔法...")

# 1. 强化正向提示词（加入 illustration, 2d, anime 等风格强化词）
prompt ="madias_style, a beautiful Chinese landscape, clear river, green trees, beautiful flowers, mountains, blue sky, white clouds, highly detailed"

# 2. 引入极度严格的反向提示词（彻底封杀真人写实感）
negative_prompt = "photorealistic, realistic, photography, 3d render, cg, cinematic, ugly, poorly drawn, deformed"

# 3. 施加 cross_attention_kwargs，强行拉高 LoRA 浓度 (比如调到 1.2 甚至 1.5)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=30, 
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 2.0} # ⬅️ 核心外挂：加大 LoRA 药量
).images[0]

image.save("test_result_fixed.png")
print("🎉 急救版验丹完成！请查看 test_result_fixed.png")