import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, Rotate, ColorJitter, GaussianBlur, Normalize, RandomCrop
from albumentations import ElasticTransform, GridDistortion, CLAHE, RandomGamma, CoarseDropout, RandomShadow, MotionBlur, ToFloat, CenterCrop
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image
from albumentations import Solarize
from torch.utils.data.dataloader import default_collate
from transformers import AutoImageProcessor
import albumentations as A





os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


class BBoxTeacherStudentDataset_SUN(Dataset):
    def __init__(self, dataset_csv_path, teacher_transform=None, student_transform=None, test=False, pretraining=False, drop=False, num_classes=2):
        """
        Args:
            dataset_csv_path (str): Path to the CSV file containing image paths and bounding box info.
            teacher_transform (albumentations.Compose, optional): Transformations to apply to the teacher (bbox) crop.
            student_transform (albumentations.Compose, optional): Transformations to apply to the student (entire) image.
        """
        if num_classes == 2:
            from utils import histology_to_int_dict_2classes as histology_to_int_dict
            from utils import int_to_histology_dict_2classes as int_to_histology_dict
        if num_classes == 3:
            from utils import histology_to_int_dict_3classes as histology_to_int_dict
            from utils import int_to_histology_dict_3classes as int_to_histology_dict

        self.histology_to_int_dict = histology_to_int_dict
        self.int_to_histology_dict = int_to_histology_dict
        
        self.dataset = pd.read_csv(dataset_csv_path)
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform
        self.pretrain = pretraining
        self.drop = drop

        # print number of elements for each class converted to int
        print(self.dataset['histology'].map(self.histology_to_int_dict).value_counts())

        if self.pretrain:
            #drop the negative class
            self.dataset = self.dataset[self.dataset['histology'] != 'Negative']

        # if drop, map unique cases linked to the same histology, then drop the first 2 cases
        # of each histology
        if drop:
            # if drop, map unique cases linked to the same histology, then drop the first 2 cases
            # of each histology
            
            unique_cases_and_histology = self.dataset[['case', 'histology']].drop_duplicates()

            # drop the first 2 cases of each histology
            unique_cases_and_histology = unique_cases_and_histology.groupby('histology').head(2)

            # remove rows with cases in unique_cases_and_histology from the dataset
            self.dataset = self.dataset[~self.dataset['case'].isin(unique_cases_and_histology['case'])]

        # drop the negative class
        self.dataset = self.dataset[self.dataset['histology'] != 'Negative']


        








        
       
        # Although we define an initial center crop transform below,
        # we will compute the crop offsets manually to adjust the bbox.
        self.initial_center_crop = A.CenterCrop(height=540, width=620)

        # Default teacher transform (lower augmentation intensity).
        # This pipeline ensures the final output size is 540×620.
        if self.teacher_transform is None:
            if test:
                self.teacher_transform = A.Compose([
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.teacher_transform = A.Compose([
                    A.Resize(height=224, width=224),
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])

        # Default student transform (stronger augmentation including Cutout).
        # This pipeline also resizes the final output to 540×620.
        if self.student_transform is None:
            if test:
                self.student_transform = A.Compose([
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.student_transform = A.Compose([
                    A.Resize(height=224, width=224),  # Resize to fixed dimensions
                    A.HorizontalFlip(p=0.5),  # Flip horizontally
                    A.VerticalFlip(p=0.5),  # Flip vertically
                    A.RandomRotate90(p=0.5),  # Random 90-degree rotations
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),  # Gaussian blur
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Elastic deformation # TESTED
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),  # Add random shadows # TESTED
                    A.MotionBlur(blur_limit=5, p=0.2),  # Motion blur # TESTED
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
                    ToTensorV2(),  # Convert to PyTorch tensor
                ])

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        # number of classes is the number of unique numbers in the dictionary

        return len(self.int_to_histology_dict)
    
    #add attribute class_elements
    def class_elements(self):
        return self.dataset['histology'].value_counts().to_dict()
    
    
    def compute_class_weights(self):
        """
        Compute class weights based on the frequency of each class in the dataset.

        Returns:
            torch.Tensor: Class weights.
        """
        histology_counts = self.dataset['histology'].map(self.histology_to_int_dict).value_counts()
        total_samples = len(self.dataset)
        num_classes = self.num_classes()

        # Compute weights: total_samples / (num_classes * class_count)
        class_weights = total_samples / (num_classes * histology_counts)
        class_weights = class_weights.sort_index()  # Ensure weights are in the correct order
        return torch.tensor(class_weights.values, dtype=torch.float32)

    def __getitem__(self, idx):
        # Read a row from the CSV.
        row = self.dataset.iloc[idx]
        image_path = os.getcwd(), row['image_path']
        
         # Replace backslashes with forward slashes
        image_path = os.path.normpath(image_path).replace('\\', '/')

        histology = self.dataset.iloc[idx]['histology']

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error reading image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error processing image at {image_path}: {e}")
            return None  # Skip this image
        
        # Compute original image dimensions.
        orig_h, orig_w, _ = image.shape

        # Calculate top-left coordinates for a center crop of size 540x620.
        crop_top = (orig_h - 540) // 2
        crop_left = (orig_w - 620) // 2

        # Step 1: Center crop the image manually.
        center_cropped_image = image

            

        # Step 2: Extract and adjust bounding box coordinates.
        # The CSV should contain: 'x', 'y', 'width', 'height',
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['width'])
        h = int(row['height'])

        x_original = x
        y_original = y
        w_original = w
        h_original = h

        

        # Adjust the bbox coordinates relative to the center crop.
        adjusted_x = x 
        adjusted_y = y 

        

        # Extract the teacher crop using the adjusted bounding box coordinates.
        if self.pretrain:
            teacher_crop = center_cropped_image[adjusted_y:adjusted_y + h, adjusted_x:adjusted_x + w]
        else:
            teacher_crop = center_cropped_image

        if teacher_crop.size == 0:
            print(f'Crop left: {crop_left}, Crop top: {crop_top}')
            print(f'Original image size:', image.shape)
            print(f'teacher crop size:', teacher_crop.shape)
            print(f'Original bbox coordinates: {x_original}, {y_original}, {w_original}, {h_original}')
            print(f'Bounding box coordinates: {adjusted_x}, {adjusted_y}, {w}, {h}')
            raise ValueError(f"Empty crop for image at {image_path}")

        # Step 3: Apply augmentations.
        # The teacher augmentation pipeline is applied to the extracted bbox region.
        teacher_image = self.teacher_transform(image=teacher_crop)['image']
        # The student augmentation pipeline is applied to the entire center-cropped image.
        student_image = self.student_transform(image=center_cropped_image)['image']

        # create the label containing the paris classification as a vector
        histology_label = np.zeros(len(self.int_to_histology_dict))
        histology_label[self.histology_to_int_dict[histology]] = 1

        scale_x = 224 / orig_w
        scale_y = 224 / orig_h

        scaled_x1 = x * scale_x
        scaled_y1 = y * scale_y
        scaled_x2 = (x + w) * scale_x
        scaled_y2 = (y + h) * scale_y

        bbox = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

        

        

        return teacher_image, student_image, histology_label, histology, image_path, scaled_x1, scaled_y1, scaled_x2, scaled_y2


if __name__=='__main__':
    dataset = BBoxTeacherStudentDataset_SUN(dataset_csv_path='data/sun_preprocessed_dataset.csv', pretraining=True, drop=True, num_classes=3)
    print('Number of classes:', dataset.num_classes())
    print('Class elements:', dataset.class_elements())
    print('Class weights:', dataset.compute_class_weights())
    for i in range(len(dataset)):
        teacher_img, student_img, histology_label, histology, image_path, sx1, sy1, sx2, sy2 = dataset[i]
        print(f'Image path: {image_path}')
        print(f'Histology: {histology}, Label: {histology_label}')
        print(f'Scaled BBox: {sx1}, {sy1}, {sx2}, {sy2}')
        print(f'Teacher image shape: {teacher_img.shape}, Student image shape: {student_img.shape}')
        if i == 5:
            break





        




    