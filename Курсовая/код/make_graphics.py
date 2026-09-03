import torch
import numpy as np
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import VGG16_Weights
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import os

# 1. Define test dataset without augmentations
class TestDS(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.files = [fn for fn in os.listdir(img_dir) if fn.endswith(('.jpg','.png'))]
        w = VGG16_Weights.DEFAULT
        self.transform = T.Compose([
            T.Resize((300,300)),
            T.ToTensor(),
            T.Normalize(mean=w.transforms().mean, std=w.transforms().std)
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        img_path = f"{self.img_dir}/{fn}"
        lbl_path = f"{self.lbl_dir}/{fn.rsplit('.',1)[0]}.txt"
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        img_t = self.transform(img)
        boxes, labels = [], []
        for line in open(lbl_path):
            c, xc, yc, ww, hh = map(float, line.split())
            x1 = (xc-ww/2)*w0; y1 = (yc-hh/2)*h0
            x2 = (xc+ww/2)*w0; y2 = (yc+hh/2)*h0
            # scale to 300
            sx, sy = 300/w0, 300/h0
            boxes.append([x1*sx, y1*sy, x2*sx, y2*sy])
            labels.append(int(c))
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return img_t, target

# IoU function
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / (boxAArea+boxBArea-interArea + 1e-6)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ssd300_vgg16(weights=None, num_classes=12, weights_backbone=VGG16_Weights.DEFAULT)
model.load_state_dict(torch.load('SSD.pth'))
model.to(device).eval()

# DataLoader
test_ds = TestDS('C:/Users/jack_/MAIN/ВУЗ/Курсовая/код/test_dataset/test_images',
                 'C:/Users/jack_/MAIN/ВУЗ/Курсовая/код/test_dataset/test_labels')
dl = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

all_true, all_pred, all_scores = [], [], []

# Inference and matching
for imgs, tgts in dl:
    imgs = [img.to(device) for img in imgs]
    with torch.no_grad():
        outs = model(imgs)
    for out, tgt in zip(outs, tgts):
        pred_boxes = out['boxes'].cpu().numpy()
        pred_labels = out['labels'].cpu().numpy()
        pred_scores = out['scores'].cpu().numpy()
        gt_boxes = tgt['boxes'].numpy()
        gt_labels = tgt['labels'].numpy()
        matched_pred = set()
        # match GT to preds
        for gt_box, gt_lbl in zip(gt_boxes, gt_labels):
            best_i, best_iou = -1, 0
            for i, pb in enumerate(pred_boxes):
                if i in matched_pred: continue
                iou_val = iou(gt_box, pb)
                if iou_val > best_iou:
                    best_iou, best_i = iou_val, i
            if best_iou >= 0.5:
                all_true.append(gt_lbl)
                all_pred.append(pred_labels[best_i])
                all_scores.append(pred_scores[best_i])
                matched_pred.add(best_i)
            else:
                # missed detection => record as false negative
                all_true.append(gt_lbl)
                all_pred.append(-1)
                all_scores.append(0.0)
        # optional: record false positives as predictions with no gt
        # for i in range(len(pred_boxes)):
        #     if i not in matched_pred:
        #         all_true.append(-1)
        #         all_pred.append(pred_labels[i])
        #         all_scores.append(pred_scores[i])

# 1. Confusion matrix
cm = confusion_matrix(all_true, all_pred, labels=list(range(12)))
plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 2. Precision-Recall and ROC-AUC per class
for cls in range(12):
    y_true = [1 if t == cls else 0 for t in all_true]
    y_score = [s if p == cls else 0 for p, s in zip(all_pred, all_scores)]
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC={auc:.2f})')
    plt.title(f'Precision-Recall for class {cls}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    print(f'Class {cls} ROC-AUC: {auc:.3f}')
