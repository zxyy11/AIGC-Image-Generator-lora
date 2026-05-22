import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ================= 1. 加载模型与权重 =================
print("正在初始化 Web 引擎、底模与 ControlNet...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# A. 加载 ControlNet 引擎 (自动下载 Canny 边缘控制模型)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=dtype
)

# B. 加载主模型 (Counterfeit-V2.5)，并把 ControlNet 外挂进去
model_id = "gsdf/Counterfeit-V2.5"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, 
    controlnet=controlnet, 
    torch_dtype=dtype
)

#加载国风水彩画风LoRA
print("🔌 正在挂载国风水彩画风 LoRA...")
lora_path = "./output_lora/madias_style_lora" 
if os.path.exists(lora_path):
    pipe.load_lora_weights(lora_path)
else:
    pipe.load_lora_weights("./madias_style_lora")

pipe.to(device)

# ================= 2. 定义预处理与生成逻辑 =================
def process_image(input_image, prompt, negative_prompt, steps, cfg_scale, lora_scale, control_scale):
    if input_image is None:
        # 如果用户没上传图，强行拦住，防止报错
        return None, None
        
    # 1. 预处理：用 OpenCV 把彩色图片变成“黑底白线”的线稿
    image = np.array(input_image)
    image = cv2.Canny(image, 100, 200) # 提取边缘
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # 2. 强制加上你的专属画风触发词
    full_prompt = f"madias_style, {prompt}"
    
    # 3. 最终生成：包含提示词、LoRA 和 线稿结构
    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        cross_attention_kwargs={"scale": lora_scale},
        controlnet_conditioning_scale=control_scale,
        control_guidance_end=0.5  # 因为过于模仿原图：让 ControlNet 在 50% 进度时关闭！给水彩留出晕染空间
    ).images[0]
    return result, canny_image

# ================= 3. 构建 Gradio 网页 UI =================
default_prompt = "scenery, outdoors, river, mountain, tree, blue sky, white clouds, no humans, traditional media, watercolor (medium), gufeng, pastel colors, green theme, highly detailed"
default_negative = "photorealistic, realistic, photography, 3d render, humans, people, ugly, blurry, bad anatomy"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("结构控制型水彩引擎")
    #gr.Markdown("基于 Counterfeit-V2.5 + 专属 LoRA + ControlNet (Canny)打造。上传一张照片，它将保留轮廓并转化为国风水彩。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("🎨 画面结构控制区")
            # 新增：参考图上传区
            input_img = gr.Image(type="pil", label="1. 上传结构参考图 (照片或现成线稿)")
            
            prompt_in = gr.Textbox(label="2. 正向提示词 (你想把这张图变成什么样？)", lines=3, value=default_prompt)
            negative_in = gr.Textbox(label="反向提示词", lines=1, value=default_negative)
            
            with gr.Accordion("进阶参数", open=False):
                steps_slider = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="迭代步数 (Steps)")
                cfg_slider = gr.Slider(minimum=1.0, maximum=15.0, value=9.5, step=0.5, label="提示词服从度 (CFG Scale)")
                lora_slider = gr.Slider(minimum=0.0, maximum=1.5, value=1.5, step=0.1, label="画风浓度 (LoRA)")
                # 新增：控制网强度滑块
                control_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.4, step=0.1, label="结构控制强度 (ControlNet)")
            
            generate_btn = gr.Button(" 提取结构并生成", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("#生成结果")
            # 新增：预览机器提炼的线稿
            edge_out = gr.Image(label="机器看到的结构线稿 (Control Map)")
            image_out = gr.Image(label="最终水彩原画")

    generate_btn.click(
        fn=process_image,
        inputs=[input_img, prompt_in, negative_in, steps_slider, cfg_slider, lora_slider, control_slider],
        outputs=[image_out, edge_out]
    )

if __name__ == "__main__":
    print("🎉 Web 服务已启动！请点击下方的 http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)