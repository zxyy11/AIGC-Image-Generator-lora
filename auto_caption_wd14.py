import os
# --- 破壁魔法开始 ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# --- 破壁魔法结束 ---

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ================= 1. 打标参数配置 =================
DATA_DIR = "./dataset/image"
# 我们使用开源界极其稳定、专攻二次元的 WD14 v2 模型
REPO_ID = "SmilingWolf/wd-v1-4-vit-tagger-v2"
THRESHOLD = 0.35 # 置信度阈值（分数大于 0.35 的标签才保留）

# 你的专属灵魂前缀和后缀（这就是那 20% 的人工灵魂注入）
PREFIX_TAGS = "madias_style"
SUFFIX_TAGS = "traditional media, watercolor (medium), gufeng, pastel colors"

# ================= 2. 自动拉取模型 =================
print("⏳ 正在连接国内镜像，拉取 WD14 自动打标大脑...")
print("（如果是第一次运行，会下载约 300MB 的模型，请耐心等待）")
csv_path = hf_hub_download(repo_id=REPO_ID, filename="selected_tags.csv")
onnx_path = hf_hub_download(repo_id=REPO_ID, filename="model.onnx")

df = pd.read_csv(csv_path)
tag_names = df["name"].tolist()

# ================= 3. 组装推断引擎 =================
# 使用 CPU 运行，防止弄脏你极其珍贵的 GPU 炼丹环境
session = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def prepare_image(image_path):
    """WD14 专用的图片预处理逻辑"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((448, 448), Image.Resampling.BICUBIC)
    # 转换为 BGR 格式的 numpy 数组，这是该模型的专属口味
    img_array = np.array(img, dtype=np.float32)[:, :, ::-1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================= 4. 启动流水线 =================
print("🚀 引擎就绪！开始智能打标...")

for root, dirs, files in os.walk(DATA_DIR):
    for file in tqdm(files, desc="打标进度"):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            img_path = os.path.join(root, file)
            txt_path = os.path.splitext(img_path)[0] + '.txt'

            # 🔒 【核心防覆盖锁】如果 txt 已经存在，说明是老图，直接跳过！
            if os.path.exists(txt_path):
                continue

            try:
                # 1. 图像送入模型
                img_tensor = prepare_image(img_path)
                probs = session.run(None, {input_name: img_tensor})[0][0]

                # 2. 提取出高置信度的二次元标签
                found_tags = []
                for i, p in enumerate(probs):
                    if p >= THRESHOLD:
                        tag_name = tag_names[i]
                        # 过滤掉系统自带的安全评级词汇
                        if tag_name not in ["safe", "questionable", "explicit", "sensitive"]:
                            # Danbooru 标准规范：把标签里的下划线替换成空格
                            found_tags.append(tag_name.replace("_", " "))

                # 3. 组装终极公式：前缀 + AI提取词 + 后缀
                final_tags_list = [PREFIX_TAGS] + found_tags + [SUFFIX_TAGS]
                final_tags_str = ", ".join(final_tags_list)

                # 4. 生成专属 TXT 文件
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_tags_str)

            except Exception as e:
                print(f"\n⚠️ 处理 {file} 时遇到小问题: {e}")

print("🎉 全部处理完毕！你的完美数据集已经诞生，快去文件夹里检查战果吧！")