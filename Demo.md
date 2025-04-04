# Installtion

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create --name py310 python=3.10
conda activate py310
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"

pip install pixeltable
git clone https://huggingface.co/spaces/svjack/Text-image-similarity-search-on-video-frames-embedding-indexes
cd Text-image-similarity-search-on-video-frames-embedding-indexes
pip install -r requirements.txt
pip install transformers==4.49.0
python app.py

git clone https://huggingface.co/datasets/svjack/Bakemonogatari_Videos_Splited_Captioned
mkdir Bakemonogatari_Videos_Splited_Captioned_Total
cp -r Bakemonogatari_Videos_Splited_Captioned/video0/* Bakemonogatari_Videos_Splited_Captioned_Total
cp -r Bakemonogatari_Videos_Splited_Captioned/video00/* Bakemonogatari_Videos_Splited_Captioned_Total
```

# Person
```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip
from pixeltable.iterators import FrameIterator
import PIL.Image

import pathlib
import pandas as pd
import numpy as np

mp4_list = pd.Series(list(pathlib.Path("Bakemonogatari_Videos_Splited_Captioned_Total/").rglob("*.mp4"))).map(str).values.tolist()
video_table = pxt.create_table('videos', {'video': pxt.Video})
video_table.insert(pd.Series(mp4_list).map(lambda x: {"video": x}).values.tolist())

'''
frames_view = pxt.create_view(
    'frames', video_table, iterator=FrameIterator.create(video=video_table.video))
'''
'''
video_table = pxt.get_table("videos")
'''
frames_view = pxt.create_view(
    'frames_numf_1', video_table, iterator=FrameIterator.create(video = video_table.video, num_frames = 1))

# Create an index on the 'frame' column that allows text and image search
frames_view.add_embedding_index('frame', image_embed=clip.using(model_id = 'openai/clip-vit-base-patch32'))

# Now we will retrieve images based on a sample image
sample_image = '战场原.webp'
from PIL import Image
sample_image = Image.open(sample_image)

sim = frames_view.frame.similarity(sample_image)
frames_view.order_by(sim, asc=False).limit(5).select(frames_view.frame, sim=sim).collect()

sample_image = '战场原.webp'
from PIL import Image
sample_image = Image.open(sample_image)

sim = frames_view.frame.similarity(sample_image)
out = frames_view.order_by(sim, asc=False).limit(5).select(frames_view.frame, frames_view.video, sim=sim).collect()
out.to_pandas()["video"].values.tolist()
```

# Landscape
```python
# Create an index on the 'frame' column that allows text and image search
frames_view.add_embedding_index('frame', 
                                string_embed=clip.using(model_id = 'openai/clip-vit-base-patch32'),
                                image_embed=clip.using(model_id = 'openai/clip-vit-base-patch32'))
frames_view

#### landscape with no one / big eye
sim = frames_view.frame.similarity("landscape with no one")
out = frames_view.order_by(sim, asc=False).limit(50).select(frames_view.frame, frames_view.video, sim=sim).collect()
out


sim = frames_view.frame.similarity("big eye")
out = frames_view.order_by(sim, asc=False).limit(250).select(frames_view.frame, frames_view.video, sim=sim).collect()
out
import os 
import shutil
out_df = out.to_pandas()

#### pip install moviepy==1.0.3 scikit-image

#### 0-1 值越小差异越小
from moviepy.editor import VideoFileClip
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_frame_difference(video_path):
    # 加载视频
    clip = VideoFileClip(video_path)
    
    # 获取第一帧和最后一帧
    first_frame = clip.get_frame(0)
    try:
        last_frame = clip.get_frame(clip.duration - 0.01)  # 稍微提前一点以确保获取到最后一帧
    except:
        print("err: ", video_path)
        return "error"
    
    # 将帧转换为灰度图像
    first_frame_gray = np.mean(first_frame, axis=2)
    last_frame_gray = np.mean(last_frame, axis=2)
    
    # 计算两帧之间的结构相似性 (SSIM)，并指定 data_range
    data_range = 1.0  # 因为 np.mean 的结果是归一化的浮点数，范围是 0-1
    ssim_value, _ = ssim(first_frame_gray, last_frame_gray, full=True, data_range=data_range)
    
    # 将 SSIM 值映射到 0-1 之间 (SSIM 已经在 0-1 之间，因此可以直接使用)
    difference = 1 - ssim_value
    
    return difference

# 示例使用
video_path = "Bakemonogatari E05___278.mp4"
difference = calculate_frame_difference(video_path)
print(f"第一帧和最后一帧的差异值: {difference}")

out_df["diff_score"] = out_df["video"].map(calculate_frame_difference)

out_df[
out_df.apply(lambda x: x["diff_score"] != "error" and x["diff_score"] < 0.5, axis = 1)
].sort_values("diff_score", ascending = False)

out_df = out_df[
out_df.apply(lambda x: x["diff_score"] != "error" and x["diff_score"] < 0.5, axis = 1)
].sort_values("diff_score", ascending = False)

os.makedirs("eye", exist_ok=True)
for i, r in out_df.iterrows():
    shutil.copy2(r["video"], os.path.join("eye", r["video"].split("/")[-1]))
    shutil.copy2(r["video"].replace(".mp4", ".txt"), os.path.join("eye", r["video"].split("/")[-1]).replace(".mp4", ".txt"))

import pathlib
import pandas as pd

def r_func(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_metadata(input_dir):
    # 创建Path对象并标准化路径
    input_path = pathlib.Path(input_dir).resolve()
    
    # 收集所有视频和文本文件
    file_list = []
    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in ('.mp4', '.txt'):
            file_list.append({
                "stem": file_path.stem,
                "path": file_path,
                "type": "video" if file_path.suffix.lower() == '.mp4' else "text"
            })
    
    # 创建DataFrame并分组处理
    df = pd.DataFrame(file_list)
    grouped = df.groupby('stem')
    
    metadata = []
    for stem, group in grouped:
        # 获取组内文件
        videos = group[group['type'] == 'video']
        texts = group[group['type'] == 'text']
        
        # 确保每组有且只有一个视频和一个文本文件
        if len(videos) == 1 and len(texts) == 1:
            video_path = videos.iloc[0]['path']
            text_path = texts.iloc[0]['path']
            
            metadata.append({
                "file_name": video_path.name,  # 自动处理不同系统的文件名
                "prompt": r_func(text_path)
            })
    
    # 保存结果到CSV
    output_path = input_path.parent / "metadata.csv"
    pd.DataFrame(metadata).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Metadata generated at: {output_path}")

!mv eye Akiyuki_Shinbo_Bakemonogatari_Eye_Style_Videos_Captiond

'''
---
configs:
- config_name: default
  data_files:
  - split: train
    path: 
    - "*.mp4"
    - "metadata.csv"
---
'''

generate_metadata("Akiyuki_Shinbo_Bakemonogatari_Eye_Style_Videos_Captiond")
```
