import os
import random
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import random


class COCODataset(Dataset):
    def __init__(self, data_dir):
        self.coco = COCO(os.path.join(data_dir, 'annotations', 'captions_train2017.json'))
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.pairs = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                self.pairs.append((img_id, ann['caption']))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_id, caption = self.pairs[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, 'train2017', img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, caption