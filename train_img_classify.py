import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"
import timm
import cv2
import random
import json
import datetime
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 使用当前时间作为名称创建 SummaryWriter
writer = SummaryWriter(log_dir=f"runs/logs/{current_time}")


torch.manual_seed(42) # 固定随机种子
random.seed(42)  # 

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir,'images')
        self.label_dir = os.path.join(data_dir,'labels')
        # get model specific transforms (normalization, resize)
        self.transform = timm.data.create_transform(is_training=False)
        if  len(os.listdir(self.img_dir)) ==  len(os.listdir(self.label_dir)):
            self.file_list = sorted(os.listdir(self.img_dir))
        else:
            print('错误，数据不匹配')
        self.label_all = json.load(open('labels_.json'))
        self.display_dict = json.load(open('label_display.json'))
        self.display_dict = {val: key for key, val in self.display_dict.items()} # 翻转key value
        self.viewpoint_dict = json.load(open('label_viewpoint.json'))
        self.viewpoint_dict = {val: key for key, val in self.viewpoint_dict.items()} # 翻转key value
        self.label_dict = json.load(open('label_label.json'))
        self.label_dict = {val: key for key, val in self.label_dict.items()} # 翻转key value
    
    # 将yolo格式的数据转化为坐标点
    def yolo2box(self,img_shape, box):
        h, w, _ = img_shape
        x_, y_, w_, h_ = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        x1 = w * x_ - 0.5 * w * w_
        x2 = w * x_ + 0.5 * w * w_
        y1 = h * y_ - 0.5 * h * h_
        y2 = h * y_ + 0.5 * h * h_
        return int(x1), int(y1), int(x2), int(y2)

    def __len__(self):
        return len(self.file_list)
        # return 512*10

    def __getitem__(self, idx):
        while 1:
            try:
                img_name = os.path.join(self.img_dir, self.file_list[idx])
                img = cv2.imread(img_name)[:,:,::-1] # 原图读取
                label_path = img_name.replace('images','labels').replace('jpg','txt').replace('png','txt') # label_txt地址
                label = random.choice(open(label_path).readlines()).replace('\n','').split(' ')
                label, box = label[0],label[1:]

                # box切割
                box = self.yolo2box(img.shape,box)
                # img_part = img[box[1]:box[3],box[0]:box[2],:] # 根据box切割图片
                img_part = img[int(box[1]*0.8):int(box[3]*1.2),box[0]:box[2],:] # 扩大上下范围
                img = Image.fromarray(img_part)
                img = self.transform(img)

                # label处理
                label = self.label_all[label]
                display,viewpoint,label = label.split('-')
                display = int(self.display_dict[display])
                viewpoint = int(self.viewpoint_dict[viewpoint])
                label = int(self.label_dict[label])
                return img, (display,viewpoint,label)
            except:
                idx = random.randint(0, len(self.file_list) - 1)
                print(img_name, '图片数据错误')


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


def calculate_f1(predictions, targets, num_classes):
    predictions = predictions.argmax(1)

    f1_scores = []
    precisions = []
    recalls = []

    for cls in range(num_classes):
        true_positives = ((predictions == cls) & (targets == cls)).sum().item()
        false_positives = ((predictions == cls) & (targets != cls)).sum().item()
        false_negatives = ((predictions != cls) & (targets == cls)).sum().item()

        precision = true_positives / (true_positives + false_positives + 1e-9)  # 避免除以零
        recall = true_positives / (true_positives + false_negatives + 1e-9)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    return avg_f1, avg_precision, avg_recall



if __name__ == "__main__":
    model = MobileNetV4().cuda()
    # model.load_state_dict(torch.load('models/img_classify/model_20148_0.66_0.76_0.91.pth')) # 加载模型权重
    
    # 多卡运行
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    data_dir = "test"
    dataset = CustomDataset(data_dir)
    
    # 切分测试训练集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.9 * dataset_size)  # 80% training, 20% testing
    train_indices, test_indices = indices[:split], indices[split:]
    with open('test.txt','w') as f:
        f.write(str(test_indices))

    global_step = 0
    step = 0
    while 1:
        train_sampler = SubsetRandomSampler(train_indices)
        dataloader = tqdm(DataLoader(dataset, sampler=train_sampler, num_workers=32, batch_size=512) ,desc='training')
        for img, labels_all in dataloader:
            img = img.cuda()
            display_y, viewpoint_y, labels_y = labels_all
            display_y, viewpoint_y, labels_y = display_y.cuda(), viewpoint_y.cuda(), labels_y.cuda()

            labels, viewpoint, display = model(img)
            loss_labels = F.cross_entropy(labels,labels_y)
            loss_viewpoint = F.cross_entropy(viewpoint,viewpoint_y)
            loss_display = F.cross_entropy(display,display_y)
            loss = loss_labels+loss_viewpoint+loss_display

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step%10 == 0:
                # 计算评价指标
                f1_label, precision_label, recall_label = calculate_f1(labels, labels_y, 23)
                f1_viewpoint, precision_viewpoint, recall_viewpoint = calculate_f1(viewpoint,viewpoint_y, 4)
                f1_display, precision_display, recall_display = calculate_f1(display,display_y, 2)

                # 在计算评价指标后，将它们写入 TensorBoard
                writer.add_scalar('F1/f1_label', f1_label, global_step)
                writer.add_scalar('Precision/precision_label', precision_label, global_step)
                writer.add_scalar('Recall/recall_label', recall_label, global_step)

                writer.add_scalar('F1/f1_viewpoint', f1_viewpoint, global_step)
                writer.add_scalar('Precision/precision_viewpoint', precision_viewpoint, global_step)
                writer.add_scalar('Recall/recall_viewpoint', recall_viewpoint, global_step)

                writer.add_scalar('F1/f1_display', f1_display, global_step)
                writer.add_scalar('Precision/precision_display', precision_display, global_step)
                writer.add_scalar('Recall/recall_display', recall_display, global_step)

                # 写入损失
                writer.add_scalar('Loss/total_loss', loss.item(), global_step)
                writer.add_scalar('Loss/loss_labels', loss_labels.item(), global_step)
                writer.add_scalar('Loss/loss_viewpoint', loss_viewpoint.item(), global_step)
                writer.add_scalar('Loss/loss_display', loss_display.item(), global_step)

                dataloader.set_description(f'f1_labels:{f1_label:.4f}, f1_viewpoint:{f1_viewpoint:.4f}, f1_display:{f1_display:.4f}, loss_labels:{loss_labels:.4f}, loss_viewpoint:{loss_viewpoint:.4f}, loss_display:{loss_display:.4f}')

                # # 保存模型
                # torch.save(model.state_dict(), f'models/img_classify/model_{f1_label:.2f}_{f1_viewpoint:.2f}_{f1_display:.2f}.pth')

            global_step+=1
        

        # 评价
        test_sampler = SubsetRandomSampler(test_indices)
        dataloader_val = tqdm(DataLoader(dataset, sampler=test_sampler, num_workers=32, batch_size=512) ,desc='evaling')
        for img, labels_all in dataloader_val:
            img = img.cuda()
            display_y, viewpoint_y, labels_y = labels_all
            display_y, viewpoint_y, labels_y = display_y.cuda(), viewpoint_y.cuda(), labels_y.cuda()
            with torch.no_grad():
                labels, viewpoint, display = model(img)
            loss_labels = F.cross_entropy(labels,labels_y)
            loss_viewpoint = F.cross_entropy(viewpoint,viewpoint_y)
            loss_display = F.cross_entropy(display,display_y)
            loss = loss_labels+loss_viewpoint+loss_display

            # 计算评价指标
            f1_label, precision_label, recall_label = calculate_f1(labels, labels_y, 23)
            f1_viewpoint, precision_viewpoint, recall_viewpoint = calculate_f1(viewpoint,viewpoint_y, 4)
            f1_display, precision_display, recall_display = calculate_f1(display,display_y, 2)

            # 在计算评价指标后，将它们写入 TensorBoard
            writer.add_scalar('F1/f1_label_val', f1_label, step)
            writer.add_scalar('Precision/precision_label_val', precision_label, step)
            writer.add_scalar('Recall/recall_label_val', recall_label, step)

            writer.add_scalar('F1/f1_viewpoint_val', f1_viewpoint, step)
            writer.add_scalar('Precision/precision_viewpoint_val', precision_viewpoint, step)
            writer.add_scalar('Recall/recall_viewpoint_val', recall_viewpoint, step)

            writer.add_scalar('F1/f1_display_val', f1_display, step)
            writer.add_scalar('Precision/precision_display_val', precision_display, step)
            writer.add_scalar('Recall/recall_display_val', recall_display, step)

            # 写入损失
            writer.add_scalar('Loss/total_loss_val', loss.item(), step)
            writer.add_scalar('Loss/loss_labels_val', loss_labels.item(), step)
            writer.add_scalar('Loss/loss_viewpoint_val', loss_viewpoint.item(), step)
            writer.add_scalar('Loss/loss_display_val', loss_display.item(), step)

            dataloader.set_description(f'val: f1_labels:{f1_label:.4f}, f1_viewpoint:{f1_viewpoint:.4f}, f1_display:{f1_display:.4f}, loss_labels:{loss_labels:.4f}, loss_viewpoint:{loss_viewpoint:.4f}, loss_display:{loss_display:.4f}')
            step+=1
        
        # 保存模型
        torch.save(model.state_dict(), f'models/model_{step}_{f1_label:.2f}_{f1_viewpoint:.2f}_{f1_display:.2f}.pth')
            