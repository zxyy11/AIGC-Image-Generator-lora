import os
# --- 破壁魔法：如果你在本地运行，保留这行；如果部署到 Hugging Face，这行会自动被忽略 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# ================= 1. 加载模型与权重 =================
print("⏳ 正在初始化 Web 引擎与底模...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 强制使用 Counterfeit 二次元神模
model_id = "gsdf/Counterfeit-V2.5"
# 如果是 CPU 运行（比如免费的云端），需要用 float32；如果有 3090，用 float16 提速
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

print("🔌 正在挂载专属玛德亚画风 LoRA...")
# 注意这里的路径：如果是本地测试，指向 output_lora；如果是 Hugging Face 部署，确保路径正确
lora_path = "./output_lora/madias_style_lora" 
if os.path.exists(lora_path):
    pipe.load_lora_weights(lora_path)
else:
    # 兼容 Hugging Face 根目录直连模式
    pipe.load_lora_weights("./madias_style_lora")

pipe.to(device)

# ================= 2. 定义核心生成逻辑 =================
def generate_image(prompt, negative_prompt, steps, cfg_scale, lora_scale):
    # 强制在提示词最前面加上你的触发词
    full_prompt = f"madias_style, {prompt}"
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        cross_attention_kwargs={"scale": lora_scale}
    ).images[0]
    
    return image

# ================= 3. 构建精美的 Gradio 网页 UI =================
# 预设好你的神级画风修饰词，省得每次手打
default_prompt = "a beautiful Chinese landscape, clear river, green trees, beautiful flowers, mountains, blue sky, white clouds, highly detailed"
default_negative = "photorealistic, realistic, photography, 3d render, cg, cinematic, ugly, poorly drawn, deformed"
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌸 玛德亚公爵府：国风水彩场景生成器")
    gr.Markdown("基于 Counterfeit-V2.5 底模与专属微调 LoRA 打造的二次元水彩风景引擎。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 画面参数控制")
            # 提示词输入框（已经贴心地去掉了 madias_style，因为代码里会自动加）
            prompt_in = gr.Textbox(label="正向提示词 (画面里要有什么？)", lines=4, value=default_prompt)
            negative_in = gr.Textbox(label="反向提示词 (不想要什么？)", lines=2, value=default_negative)
            
            # 进阶参数滑动条
            with gr.Accordion("⚙️ 进阶参数 (萌新可保持默认)", open=False):
                steps_slider = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="迭代步数 (Steps)")
                cfg_slider = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="提示词服从度 (CFG Scale)")
                lora_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.2, step=0.1, label="画风浓度 (LoRA Weight)")
            
            generate_btn = gr.Button("🚀 立即生成", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 生成结果")
            image_out = gr.Image(label="你的专属原画")

    # 绑定按钮点击事件
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_in, negative_in, steps_slider, cfg_slider, lora_slider],
        outputs=image_out
    )

# ================= 4. 启动服务器 =================
if __name__ == "__main__":
    print("🎉 Web 服务已启动！请点击下方的 http://127.0.0.1:7860 链接查看网页！")
    demo.launch(server_name="0.0.0.0", server_port=7860)