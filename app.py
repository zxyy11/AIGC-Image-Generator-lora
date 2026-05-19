import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# 1. 核心魔法：自动检测环境（有3090就用CUDA+半精度，云端免费机就用CPU+单精度）
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"🚀 当前引擎已点火，运行环境检测为: {device.upper()}")

# 2. 加载基础大模型
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

# 3. 定义生图函数
def generate_image(prompt):
    print(f"正在生成图像: {prompt}")
    # 动态调整推理步数：本地显卡跑20步保质量，云端CPU跑15步防超时报错
    steps = 20 if device == "cuda" else 15
    image = pipe(prompt=prompt, num_inference_steps=steps).images[0]
    return image

# 4. 搭建 Gradio 交互网页
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎨 AIGC 图像生成器")
    gr.Markdown(f"**当前运行节点:** `{device.upper()}` (如果是 CPU 模式，生图约需 1-2 分钟，请耐心等待)")
    
    with gr.Row():
        # 默认提示词设置为了你熟悉的场景
        default_prompt = "the grand hall of Madias Duke mansion, medieval, elegant, visual novel background, highly detailed, anime style"
        prompt_input = gr.Textbox(label="输入英文提示词 (Prompt)", value=default_prompt, lines=3)
        
    with gr.Row():
        submit_btn = gr.Button("🚀 开始生成", variant="primary")
        
    with gr.Row():
        image_output = gr.Image(label="生成的图像结果")

    # 绑定按钮点击事件
    submit_btn.click(fn=generate_image, inputs=prompt_input, outputs=image_output)

# 5. 启动应用
if __name__ == "__main__":
    demo.launch()