from os import listdir
from os.path import join
import os
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode
from torchvision.transforms import InterpolationMode

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calc_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor(),
    ])

class TrainDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(TrainDatasets, self).__init__()
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.crop_size = calc_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_img = self.hr_transform(Image.open(self.img_filenames[index]).convert('L'))
        lr_img = self.lr_transform(hr_img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.img_filenames)

class ValDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(ValDatasets, self).__init__()
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):

        hr_img = self.hr_transform(Image.open(self.img_filenames[index]).convert('L'))
        lr_img = self.lr_transform(hr_img)
        hr_restore_img = ToTensor()(Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)(ToPILImage()(lr_img)))
        return lr_img, hr_restore_img, hr_img

    def __len__(self):
        return len(self.img_filenames)

class TestDatasets:
    def __init__(self, data_dir, upscale_factor):
        super(TestDatasets, self).__init__()
        self.image_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_img_file(x)]
        if (upscale_factor == 2):
            self.lr_path = data_dir + '/LRdata_x2/'
        elif (upscale_factor == 3):
            self.lr_path = data_dir + '/LRdata_x3/'
        elif (upscale_factor == 4):
            self.lr_path = data_dir + '/LRdata_x4/'
        self.hr_path = data_dir + '/HRdata/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_img_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_img_file(x)]

    def __getitem__(self, index):

        img_name = self.lr_filenames[index].split('/')[-1]
        if 'ad' in img_name:
            label = 0
        if 'mci' in img_name:
            label = 1
        if 'nc' in img_name:
            label = 2

        if 'mild' in img_name:
            label = 0
        if 'non' in img_name:
            label = 1
        if 'verymild' in img_name:
            label = 2
        lr_img = Image.open(self.lr_filenames[index])
        w, h = lr_img.size
        hr_img = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_img)
        return img_name, ToTensor()(lr_img), ToTensor()(hr_restore_img), ToTensor()(hr_img)

    def __len__(self):
        return len(self.lr_filenames)

