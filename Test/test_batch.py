import os
from Train import pytorch_ssim
from Train.data_utils import TestDatasets
from model.DConv_ResnetSR_map import DConv_ResnetSR_map
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from math import log10
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(upscale_factor,
        model_name='',
        test_dir='',
        out_dir=''):
    upscale_factor=upscale_factor

    results = {'Data': {'psnr': [], 'ssim': []}}

    model = DConv_ResnetSR_map(upscale_factor).eval().cuda()

    model_pth = 'epoch_'+ str(upscale_factor) + '.pth'
    print(model_name + ': 测试模型')
    epochs_path=os.path.join(out_dir+ str(model_name)+'_model_file/')
    out_path = os.path.join(out_dir + str(model_name) + '_model_file/')

    test_result_dir = os.path.join(out_path, 'test_results_image/')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    net_pth=os.path.join(epochs_path, 'epochs/'+ model_pth)

    model.load_state_dict(torch.load(net_pth,map_location='cuda:0'))

    test_set = TestDatasets(test_dir, upscale_factor=upscale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing datasets]')
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = Variable(lr_image)
            hr_image = Variable(hr_image)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
            with torch.no_grad():
                sr_image, feat, x_map, y_map =model(lr_image)
            mse = ((hr_image - sr_image[0]) ** 2).data.mean()
            psnr = 10 * log10(1 / mse)
            ssim = pytorch_ssim.ssim(sr_image, hr_image)
            pic_path = os.path.join(test_result_dir + 'epochs_' + str(upscale_factor)+ '/')
            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            utils.save_image(sr_image, pic_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                             image_name.split('.')[-1], padding=5)
            results['Data']['psnr'].append(psnr)
            results['Data']['ssim'].append(ssim)

        saved_results = {'psnr': [], 'ssim': []}
        for item in results.values():
            psnr = np.array(item['psnr'])
            ssim = np.array(item['ssim'])
            if (len(psnr) == 0) or (len(ssim) == 0):
                psnr = 'No data'
                ssim = 'No data'
            else:
                sum = 0
                for item in psnr:
                    sum += item
                psnr = sum / len(psnr)
                sum = 0
                for item in ssim:
                    sum += item
                ssim = sum / len(ssim)
            saved_results['psnr'].append(psnr)
            saved_results['ssim'].append(ssim)
        data_frame = pd.DataFrame(saved_results, results.keys())
        data_frame.to_csv(test_result_dir + 'epochs_' + str(upscale_factor) + '_test_results.csv',index_label='Data')

if __name__ == '__main__':
        main(upscale_factor=4,
             model_name='DConvResnetSR_map_maploss',
             test_dir='../Datasets/Demented_MRI/SRtest',
             out_dir='../Experiment/experiment_singelDemented_MRI/'
             )
