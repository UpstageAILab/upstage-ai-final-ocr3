import albumentations as A
from augraphy import *
import os
import cv2
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image


random.seed(42)
path = "./datasets/images/train/"
image_li = os.listdir(path)

ink_phase = [

    InkBleed(
        intensity_range=(0.2, 0.4),
        kernel_size=random.choice((3, 3)),
        severity=(0.2, 0.4),
        p=0.33,
    ),
    BleedThrough(
        intensity_range=(0.1, 0.3),
        color_range=(30, 224),
        ksize=(17, 17),
        sigmaX=1,
        alpha=random.uniform(0.1, 0.2),
        offsets=(10, 20),
        p=0.5
    )
    ]

paper_phase = [
]

post_phase = [
    Jpeg(               # jpg 파일 압축 효과 재현
        quality_range=(10, 15),
        p=0.33,
    ),


    Scribbles(
        scribbles_type="lines",
        scribbles_ink="pen",
        scribbles_location="random",
        scribbles_size_range=(250, 450),
        scribbles_count_range=(2, 3),
        scribbles_thickness_range=(1, 3),
        scribbles_brightness_change=[32, 64, 128],
        scribbles_lines_stroke_count_range=(1, 6),
        p = 0.33
    ),

]
pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

from_file_path = './datasets/images/train'
to_file_path = './datasets/images/train_aug'
if os.path.exists(to_file_path):
    from distutils.dir_util import copy_tree
    copy_tree(from_file_path, to_file_path)
else:
    import shutil
    shutil.copytree(from_file_path, to_file_path)

num = 2

for a in range(num):

    for x in tqdm(image_li):
        img = cv2.imread(os.path.join(path, x))
        image_augmented = pipeline(img)
        image_augmented = Image.fromarray(image_augmented)
        image_augmented.save(f'./datasets/images/train_aug/{x.replace(".jpg",f"_aug_{a}.jpg")}','JPEG')

    import json

    with open("./datasets/jsons/train.json",'r') as f:
        text = json.load(f)

    tmp = text

    for i in tqdm([*text['images'].keys()]):
        tmp['images'][i.replace('.jpg',f'_aug_{a}.jpg')] = text['images'][i]

with open("./datasets/jsons/train_aug.json",'w') as f:
    json.dump(tmp, f)

