# 制作人脸特征向量的数据库 最后会保存两个文件，分别是数据库中的人脸特征向量和对应的名字。当然也可以保存在一起
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

from glob import glob
from PIL import Image
all_files = glob(r'.\celeba1\celeba_raw_pic\*.jpg')
#print(all_files)
# # InceptionResnetV1提供了两个预训练模型，分别在vggface数据集和casia数据集上训练的。
# # 预训练模型如果不手动下载，可能速度会很慢，可以从作者给的谷歌云链接下载，然后放到C:\Users\你的用户名\.cache\torch\checkpoints这个文件夹下面
# # 如果是linux系统，那么存放在/home/你的用户名/.cache/torch/checkpoints下面
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#
#
# dataset = datasets.ImageFolder('./database/origin')  #加载数据库
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
# aligned = []  # aligned就是从图像上抠出的人脸，大小是之前定义的image_size=160
# names = []
# i= 1
for f in all_files:
    filename = f.split("\\")[-1]
    path = r'.\celeba\face\{}'.format(filename)  # 这个是要保存的人脸路径
    img = Image.open(f)
    x_aligned, prob = mtcnn(img, return_prob=True,save_path= path)
    if x_aligned == None:
        print("file {} cannot detect face".format(filename))
#
# file 000199.jpg cannot detect face
# file 001401.jpg cannot detect face
# file 002214.jpg cannot detect face
# file 002432.jpg cannot detect face
# file 002920.jpg cannot detect face
# file 003928.jpg cannot detect face
# file 003946.jpg cannot detect face
# file 004932.jpg cannot detect face
# file 005283.jpg cannot detect face
# file 006010.jpg cannot detect face
# file 006531.jpg cannot detect face
# file 007726.jpg cannot detect face
# file 008287.jpg cannot detect face
# file 011529.jpg cannot detect face
# file 011793.jpg cannot detect face
# file 013374.jpg cannot detect face
# file 013654.jpg cannot detect face
# file 014999.jpg cannot detect face
# file 016530.jpg cannot detect face
# file 016797.jpg cannot detect face
# file 017282.jpg cannot detect face
# file 017586.jpg cannot detect face
# file 018309.jpg cannot detect face
# file 018599.jpg cannot detect face
# file 018884.jpg cannot detect face
# file 019205.jpg cannot detect face
# file 019377.jpg cannot detect face

