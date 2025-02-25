import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import json
import copy
import os

class TACODataset(Dataset):
    def __init__(self, root_dir, annotation_file, class_map=None, transform=None):
        """
        Args:
            root_dir (string): 이미지가 있는 디렉토리 경로
            annotation_file (string): annotations_X_train.json 또는 annotations_X_test.json 파일 경로
            class_map (dict, optional): 기존 클래스 이름을 새로운 클래스 이름으로 매핑하는 딕셔너리
            transform (callable, optional): 이미지에 적용할 변환
        """
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        # Apply class mapping if provided
        if class_map:
            self.replace_dataset_classes(self.coco, class_map)

        # Create category mapping
        self.cat_ids = {}
        self.cat_id_to_label = {}
        for i, cat in enumerate(self.coco['categories'], 1):  # 1부터 시작 (0은 배경)
            self.cat_ids[cat['id']] = cat['name']
            self.cat_id_to_label[cat['id']] = i

        # Get all valid image ids
        self.ids = []
        for img in self.coco['images']:
            if os.path.exists(os.path.join(self.root_dir, img['file_name'])):
                self.ids.append(img['id'])

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)

        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Get annotations
        anns = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_id]

        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            # Convert category_id to sequential label
            labels.append(self.cat_id_to_label[ann['category_id']])

        # Handle images without annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    def replace_dataset_classes(self, dataset, class_map):
        """ Replaces classes of dataset based on a dictionary"""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = copy.deepcopy(dataset['categories'])
        dataset['categories'] = []
        class_ids_map = {}  # map from old id to new id

        # Assign background id 0
        has_background = False
        if 'Background' in class_new_names:
            if class_new_names.index('Background') != 0:
                class_new_names.remove('Background')
                class_new_names.insert(0, 'Background')
            has_background = True

        # Replace categories
        for id_new, class_new_name in enumerate(class_new_names):

            # Make sure id:0 is reserved for background
            id_rectified = id_new
            if not has_background:
                id_rectified += 1

            category = {
                'supercategory': '',
                'id': id_rectified,  # Background has id=0
                'name': class_new_name,
            }
            dataset['categories'].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original['name']] == class_new_name:
                    class_ids_map[class_original['id']] = id_rectified

        # Update annotations category id tag
        for ann in dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]

    @property
    def num_classes(self):
        return len(self.cat_ids) + 1  # +1 for background