import os
# 🚀 魔法指令：强制使用国内高速镜像源下载模型，解决网络超时报错！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# 1. 加载模型 (现在它会从国内镜像站极速下载了)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. 将模型推送到你的 3090 显卡上
pipe = pipe.to("cuda")

# 3. 定义生图核心函数
def generate_image(prompt):
    print(f"🚀 正在狂飙 3090 生成图像: {prompt}")
    image = pipe(prompt=prompt, num_inference_steps=20).images[0]
    return image

# 4. 搭建本地交互网页
with gr.Blocks() as demo:
    gr.Markdown("# 🏠 我的本地 AIGC 图像生成器")
    
    with gr.Row():
        default_prompt = "the grand hall of Madias Duke mansion, medieval, elegant, visual novel background, highly detailed, anime style"
        prompt_input = gr.Textbox(label="输入英文提示词 (Prompt)", value=default_prompt)
        submit_btn = gr.Button("调用 3090 生成", variant="primary")
        
    image_output = gr.Image(label="生成的图像")

    submit_btn.click(fn=generate_image, inputs=prompt_input, outputs=image_output)

demo.launch(share=True)