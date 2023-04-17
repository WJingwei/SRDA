from os import listdir
from os.path import join

from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calc_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)



def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),# range [0, 255] -> [0.0,1.0] 归一化
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
        # Resize(256),#400
        # CenterCrop(256),#400
        ToTensor(),
    ])



class TrainDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(TrainDatasets, self).__init__()
        #self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.image_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.crop_size = calc_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        image_name = self.image_filenames[index]
        if 'ad' in image_name:
            label = 0
        if 'mci' in image_name:
            label = 1
        if 'nc' in image_name:
            label = 2
        if 'no' in image_name:
            label = 0
        if 'y' in image_name:
             label = 1
        if 'mild' in image_name:
            label = 0
        if 'non' in image_name:
            label = 1
        if 'verymild' in image_name:
            label = 2
        hr_img = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_img = self.lr_transform(hr_img)
        label=int(label)
        return lr_img, hr_img,label

    def __len__(self):
        return len(self.image_filenames)



class ValDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(ValDatasets, self).__init__()
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        image_name = self.img_filenames[index]
        if 'ad' in image_name:
            label = 0
        if 'mci' in image_name:
            label = 1
        if 'nc' in image_name:
            label = 2
        if 'mild' in image_name:
            label = 0
        if 'non' in image_name:
            label = 1
        if 'verymild' in image_name:
            label = 2
        if 'no' in image_name:
            label = 0
        if 'y' in image_name:
            label = 1
        hr_img = self.hr_transform(Image.open(self.img_filenames[index]))
        lr_img = self.lr_transform(hr_img)
        hr_restore_img = ToTensor()(Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)(ToPILImage()(lr_img)))
        return lr_img, hr_restore_img, hr_img,label

    def __len__(self):
        return len(self.img_filenames)


class TestDatasets:
    def __init__(self, data_dir, upscale_factor):
        super(TestDatasets, self).__init__()
        self.lr_path = data_dir   + '/LRdata_x4/'
        self.hr_path = data_dir + '/HRdata/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_img_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_img_file(x)]

    def __getitem__(self, index):
        image_nameId = self.image_filenames[index]
        if 'ad' in image_nameId:
            label = 0
        if 'mci' in image_nameId:
            label = 1
        if 'nc' in image_nameId:
            label = 2
        if 'mild' in image_nameId:
            label = 0
        if 'non' in image_nameId:
            label = 1
        if 'verymild' in image_nameId:
            label = 2

        img_name = self.lr_filenames[index].split('/')[-1]
        lr_img = Image.open(self.lr_filenames[index])
        w, h = lr_img.size
        hr_img = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_img)
        return img_name, ToTensor()(lr_img), ToTensor()(hr_restore_img), ToTensor()(hr_img),label

    def __len__(self):
        return len(self.lr_filenames)

# #Debug
# if __name__ == '__main__':
#
#     test = ValDatasets('Datasets/MRI/SRval/', 48, 4)
#     print(test.__getitem__(0)[0].shape)
#     print(type(test.__getitem__(0)[1]))
#     # print(test.__getitem__(0)[2].shape)
