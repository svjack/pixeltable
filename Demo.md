```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create --name py310 python=3.10
conda activate py310
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"

pip install pixeltable
git clone https://huggingface.co/spaces/svjack/Text-image-similarity-search-on-video-frames-embedding-indexes
cd Text-image-similarity-search-on-video-frames-embedding-indexes
pip install -r requirement.txt
python app.py

git clone https://huggingface.co/datasets/svjack/Bakemonogatari_Videos_Splited_Captioned
mkdir Bakemonogatari_Videos_Splited_Captioned_Total
cp -r Bakemonogatari_Videos_Splited_Captioned/video0/* Bakemonogatari_Videos_Splited_Captioned_Total
cp -r Bakemonogatari_Videos_Splited_Captioned/video00/* Bakemonogatari_Videos_Splited_Captioned_Total
```

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
sim = frames_view.frame.similarity(sample_image)
frames_view.order_by(sim, asc=False).limit(5).select(frames_view.frame, sim=sim).collect()
```
