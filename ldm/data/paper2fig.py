import os
import cv2
import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import PureWindowsPath
from random import randint
import json
import albumentations as A
import torchvision.transforms.functional as F
from torchvision import transforms
import re


from matplotlib import pyplot as plt
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imsave('test.png', img)

def get_caption(conditionings, caption_modality):
    final_caption = ""

    if caption_modality == 0:
        final_caption = conditionings['captions_galai'][0]
    # ocr recognition for conditioning
    elif caption_modality == 1:
        captions = conditionings['captions_galai']
        # The actual caption
        cap = captions[0].rstrip()
        first_caption = cap + " " if cap.endswith(".") else cap + ". "

        # References of the figure in the text
        reference_captions = ""
        for c in captions[1:]:
            c = c.rstrip()
            if not c.endswith("."): # Add a period between paragraphs
                reference_captions += c + " . "
            elif cap.endswith(". "):
                reference_captions += c
            else:
                reference_captions += c + " "

        
        if caption_modality == 0: # Only first caption
            final_caption = first_caption
        elif caption_modality == 1: # First caption and references
            final_caption = first_caption + reference_captions
    elif caption_modality == 2:
        for ocr_res in conditionings['ocr_result']['ocr_result']:
            final_caption = final_caption + ocr_res['text'] + ','

    elif caption_modality == 3:
        final_caption = conditionings['ocr_with_coords']

    return final_caption.rstrip()

# Depricated, now we use A.PadIfNeeded
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

class Paper2FigBase(Dataset):
    def __init__(self, json_file, size, caption_modality = 0, random_crop=False, square_pad = False, use_roi_bboxes=False):
        self.json_file = json_file
        self.root_dir =  os.path.dirname(json_file)
        self.size = size
        self.caption_modality = caption_modality
        self.random_crop = random_crop
        self.square_pad = square_pad
        self.use_roi_bboxes = use_roi_bboxes

        # Read json file and get the data
        with open(self.json_file) as f:    
            self.data_json = json.load(f)

        # Experiment with filtering figures by aspect ratio (avoid extreme cases where padding destroys figure information)
        self.data = []
        for figure in self.data_json:
            if figure['aspect'] >= 0.5 and figure['aspect'] <= 2:
                # if 'cnn' in figure['class_tag']:
                self.data.append(figure)
                
        del self.data_json

        # TODO: Create a single self.transform that contains the desired transform
        if self.square_pad:
            self.square_pad_transform = A.Compose([ 
                A.LongestMaxSize(max_size = self.size),
                A.PadIfNeeded(min_width=self.size, min_height=self.size, border_mode=cv2.BORDER_CONSTANT, value = [255, 255, 255])
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields = ['category_ids']))
        else:
            self.image_rescaler = A.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_AREA)
            if self.random_crop: # TODO: Mape composable and add bbox params
                self.cropper = A.RandomCrop(height=self.size, width=self.size)
            else:
                self.cropper = A.CenterCrop(height=self.size, width=self.size)

            if self.use_roi_bboxes:
                self.bbox_transform = A.Compose([ # TODO: Test this out
                    self.image_rescaler,
                    self.cropper
                ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields = ['category_ids']))


    def get_bboxes_tensor(self, ocr_result):
        bboxes = []
        for item in ocr_result:
            bbox = item['bbox']
            coords = re.findall(r'\d+[.]?\d*', bbox)
            coords_pascal_voc = [round(float(coords[0]), 2), round(float(coords[1]), 2), round(float(coords[2]), 2), round(float(coords[5]), 2)]
            bboxes.append(coords_pascal_voc)
        return bboxes

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        sample = {}

        # Image
        image_file = os.path.join(self.root_dir, 'figures', self.data[i]['figure_id']+'.png')
        sample['image_file'] = image_file
        # Bboxes for text location
        if self.use_roi_bboxes:
            bboxes = self.get_bboxes_tensor(self.data[i]['ocr_result']['ocr_result'])
            ids = [1 for i in range(len(bboxes))]

        try:
            # Read image with PIL
            image = Image.open(image_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = np.array(image).astype(np.uint8)

            # TODO: Apply only self.transform with bboxes
            if self.square_pad:
                tr_im = self.square_pad_transform(image=image, bboxes=bboxes if self.use_roi_bboxes else [], category_ids = ids if self.use_roi_bboxes else [])
                image = tr_im['image']
                sample['bboxes'] = tr_im['bboxes']
            else: # TODO define the tranfrorm to use bboxes
                image = self.image_rescaler(image=image)['image']
                image = self.cropper(image=image)['image']

            sample['image'] = (image/127.5 - 1.0).astype(np.float32)
            
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {i}")
            return self.skip_sample(i)

        # Caption
        sample['caption'] = get_caption(self.data[i], self.caption_modality)
        return sample


class Paper2FigTrain(Paper2FigBase):
    def __init__(self, **kwargs):
        self.shuffle = True # TODO: THis should be user-defined in yaml
        super().__init__(**kwargs)

class Paper2FigValidation(Paper2FigBase):
    def __init__(self, **kwargs):
        self.shuffle = False # TODO: THis should be user-defined in yaml
        super().__init__(**kwargs)
        # self.data = [self.data[0]]

from PIL import Image
import matplotlib.pyplot as plt

def plot_grid_of_images_from_path_list(path_list, rows, cols, out_path):
    images = []
    for path in path_list:
        images.append(Image.open(path))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig, axs = plt.subplots(rows, cols, figsize=(30, 30))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis('off')
    plt.savefig(out_path, format='png')