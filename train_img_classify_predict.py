import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"
import timm
import cv2
import random
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MobileNetV4(nn.Module):
    def __init__(self, labels_classes=23,viewpoint_classes=4,display_classes=2):
        super(MobileNetV4, self).__init__()
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            num_classes=0  # Remove the classifier nn.Linear
        )
        self.classifier1 = nn.Linear(1280, labels_classes)
        self.classifier2 = nn.Linear(1280, viewpoint_classes)
        self.classifier3 = nn.Linear(1280, display_classes)

    def forward(self, x):
        embedding = self.backbone(x)
        labels = self.classifier1(embedding)
        viewpoint = self.classifier2(embedding)
        display = self.classifier3(embedding)
        return labels, viewpoint, display

# 将yolo格式的数据转化为坐标点
def yolo2box(img_shape, box):
    h, w, _ = img_shape
    x_, y_, w_, h_ = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x1 = w * x_ - 0.5 * w * w_
    x2 = w * x_ + 0.5 * w * w_
    y1 = h * y_ - 0.5 * h * h_
    y2 = h * y_ + 0.5 * h * h_
    return int(x1), int(y1), int(x2), int(y2)
    
model = MobileNetV4().cuda()
# model.load_state_dict(torch.load('models/img_classify/model_20148_0.66_0.76_0.91.pth',map_location='cpu'))
model.eval()

# 图片数据准备
# img = torch.rand(2,3,224,224)
path_dir = 'test/images'
transform = timm.data.create_transform(is_training=False)
imgs = []
origin = []
labels_name = []
labels_dict = json.load(open('labels_.json'))
test_index = eval(open('test.txt').readlines()[0])
img_names = sorted(os.listdir(path_dir))
for i in random.choices(test_index,k=10):
    img_path = os.path.join(path_dir,img_names[i])
    img = cv2.imread(img_path)[:,:,::-1]

    # 获取标签
    label_path = img_path.replace('images','labels').replace('jpg','txt').replace('png','txt') # label_txt地址
    print(img_path,label_path)
    label = random.choice(open(label_path).readlines()).replace('\n','').split(' ')
    label, box = label[0],label[1:]
    labels_name.append(labels_dict[str(label)])
    box = yolo2box(img.shape, box)
    img_part = img[box[1]:box[3],box[0]:box[2],:] # 根据box切割图片
    origin.append(cv2.resize(img_part[:,:,::-1],(224,224)))

    img = Image.fromarray(img_part)
    img = transform(img)
    imgs.append(img)
img = torch.stack(imgs)

cv2.imwrite('res.png',np.concatenate(origin,1))
# 模型预测
with torch.no_grad():
    labels, viewpoint, display = model(img.cuda())

# 模型输出整理
labels = labels.argmax(1)
viewpoint = viewpoint.argmax(1)
display = display.argmax(1)
display_dict = json.load(open('label_display.json'))
viewpoint_dict = json.load(open('label_viewpoint.json'))
label_dict = json.load(open('label_label.json'))
for l,v,d,label in  zip(labels,viewpoint,display,labels_name):
    print('pred： ',display_dict[str(int(d))],viewpoint_dict[str(int(v))],label_dict[str(int(l))])
    print('label：',label)
    print('\n')