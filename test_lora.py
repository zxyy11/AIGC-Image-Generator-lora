import torch
import os
# ---国内镜像加速下载 --- 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from diffusers import StableDiffusionPipeline

print("添加 LoRA微调")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "gsdf/Counterfeit-V2.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.load_lora_weights("./output_lora/madias_style_lora")
pipe.to(device)


# 1. 正向提示词（加入 illustration, 2d, anime 等风格强化词）
prompt ="madias_style, a beautiful Chinese landscape, clear river, green trees, beautiful flowers, mountains, blue sky, white clouds, highly detailed"

# 2. 反向提示词
negative_prompt = "photorealistic, realistic, photography, 3d render, cg, cinematic, ugly, poorly drawn, deformed"

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=30, 
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 2.0} #调高 LoRA 占比
).images[0]

image.save("test_result_fixed.png")
print("测试完成！请查看 test_result_fixed.png")