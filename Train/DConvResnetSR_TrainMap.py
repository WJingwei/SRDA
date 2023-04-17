from datetime import datetime
import torch.nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import pytorch_ssim
from tqdm import tqdm
from math import log10
import pandas as pd
from data_label import TrainDatasets, ValDatasets, display_transform
from model.DConv_ResnetSR_map import DConv_ResnetSR_map
from model.MyLoss import MyLoss


def train(model_name,data_name,num_epochs,upscale_factor,train_dir='',val_dir='',save_dir='',):
    upscale_factor = upscale_factor
    batch_size = 8
    if (data_name == 'AD_PET'):
        crop_size = 64
    elif (data_name == 'AD_MRI'):
        crop_size = 128
    elif (data_name == 'Demented_MRI'):
        crop_size = 64
    model_save_dir = os.path.join(save_dir + str(model_name) + '_model_file')

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    print('model_name:',model_name)
    print('-' * 10 + '> start loading data... ', '[ {} ]'.format(datetime.now()))

    # dataloader and remember pytorch needs to write the dataloader file
    train_set = TrainDatasets(data_dir=train_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = ValDatasets(data_dir=val_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    # feed the data into DataLoader
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    print('-' * 10 + '> data loading...')
    net= DConv_ResnetSR_map(upscale_factor).cuda()
    print('-' * 10 + '>model parameters:', sum(param.numel() for param in net.parameters()))
    criterion =MyLoss().cuda()
    optimizer_net = optim.Adam(net.parameters(), lr=0.0001)
    results = {'loss': [], 'psnr': [], 'ssim': []}
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

    for epoch in range(1, num_epochs+1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0}
        net.train()
        for data, target,label in train_bar:
            if torch.cuda.is_available():
                data = data.cuda()
                target=target.cuda()
                label=label.cuda()
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                label=label.cuda()
            input_data = Variable(data)
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            sr, feat,x_map,y_map= net(input_data)
            if torch.cuda.is_available():
                sr = sr.cuda()
                feat = feat.cuda()
                x_map = x_map.cuda()
                y_map = y_map.cuda()
            loss = criterion(sr, real_img, feat, x_map, y_map, label).cuda()
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
            running_results['loss'] = loss.item() * batch_size
            train_bar.set_description(desc='[%d/%d] Loss: %.4f' %(epoch,num_epochs,running_results['loss'] / running_results['batch_sizes']))

        net.eval()
        out_path = os.path.join(model_save_dir, 'training_results/')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        results['loss'].append(running_results['loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 100 == 0:
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                val_images = []
                for val_lr, val_hr_restore, val_hr,label in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr,feat,x_map,y_map= net(lr)
                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim *batch_size
                    # psnr counting
                    valing_results['psnr'] = 10 * log10(
                        (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))
               #     extend method. val_images is a list object extend method
                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)),display_transform()(hr.data.cpu().squeeze(0)),display_transform()(sr.data.cpu().squeeze(0))])
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 9)
                val_save_bar = tqdm(val_images, desc='[saving training_results results]')
                index = 1

                for image in val_save_bar:
                    image = torchvision.utils.make_grid(image, nrow=3, padding=5)
                    torchvision.utils.save_image(image,os.path.join(out_path, 'epoch_%d_index_%d.png' % (epoch, index)),padding=5)
                    index += 1
            # save model parameters
            model_para_dir = os.path.join(model_save_dir, 'epochs/')
            if not os.path.exists(model_para_dir):
               os.mkdir(model_para_dir)
            torch.save(net.state_dict(),os.path.join(model_para_dir, 'epoch_%d_%d.pth' % (upscale_factor, epoch)))

            results['psnr'].pop()
            results['ssim'].pop()

            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])


            out_path = os.path.join(model_save_dir, 'statistics/')
            if not os.path.exists(out_path):
               os.mkdir(out_path)

            data_frame = pd.DataFrame(data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']})

            if not os.path.exists(os.path.join(out_path, 'SRF_' + str(upscale_factor))):
                os.mkdir(os.path.join(out_path, 'SRF_' + str(upscale_factor)))
            data_frame.to_csv(out_path + 'SRF_' + str(upscale_factor) + '_train_results.csv',index_label='Epoch')


if __name__ == '__main__':
    train(model_name='DConvResnetSR_map_maploss',
          data_name='Demented_MRI',
          num_epochs=2000,
          upscale_factor = 4,
          train_dir='../Datasets/Demented_MRI/SRtrain',
          val_dir='../Datasets/Demented_MRI/SRval',
          save_dir='../Experiment/experiment_singelDemented_MRI/'
          )


