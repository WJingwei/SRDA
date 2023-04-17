# PyTorch implementation of SRDA

This work take the Alzheimer’s disease diagnosis as the downstream task and propose a novel diagnosis-guided medical image SR network, which can make the SR and diagnosis be boosted by each other. 

For more information, checkout the project site [(https://github.com/WJingwei/SRDA)].

# Getting started 

## Dependencies
* Python 3.7
* torch 1.11.0+cu113
* torchvision 0.12.0
* cuda 11.3
* numpy 1.21.6
* opencv-python 4.5.5.64
* pandas 1.3.5
* scipy 1.7.3
* h5py 3.6.0
* imageio 2.17.0


## Datasets
* We used datasets to train and test our model. Please download it from [here](https://pan.baidu.com/s/14MdYprk7Lpo3A5lfzxd-FA?pwd=0718).
* For training、validation and test
    -  ADNI-MRI(This work use AD_MRI instead of this name)
    -  ADNI-PET(AD_PET)
    -  Demented-MRI(Demented_MRI)


After download all datasets, the folder ```data``` should be like this:
```
    Datasets
    ├──  AD_MRI
    │   ├──SRtest
    │   │   ├──HRdata
    │   │   ├──LRdata_x4
    │   ├──SRtrain
    │   ├──SRval
    ├──  AD_PET
    │   ├──SRtest
    │   │   ├──HRdata
    │   │   ├──LRdata_x4
    │   ├──SRtrain
    │   ├──SRval
    ├──  Demented_MRI
    │   ├──SRtest
    │   │   ├──HRdata
    │   │   ├──LRdata_x4
    │   ├──SRtrain
    │   ├──SRval
         
```

## Training
To train our model, run the following script. 
```bash
$ python DConvResnetSR_TrainMap.py
```

## Using the pretrained models
* We put the trained super-resolution model in the ``Experiment/experiment_singel+DATA_NAME/DConvResnetSR_map_maploss_model_file/epochs`` folder. <br>
* We put the trained classification model in the ``ClassficationModel/DATA_NAME`` folder. <br>
* Please download it from [here](https://pan.baidu.com/s/14MdYprk7Lpo3A5lfzxd-FA?pwd=0718).

## Evaluation
To evaluate our student model, run following script. 
```
$ python test_batch.py
```

---
## Citation

```
@inproceedings{Jingwei_2023_Icme,
    author={Jingwei Wang, Peng Zhou, Xianjun Han, Yanming Chen},
    title={Medical Image Super-Resolution via Diagnosis-Guided Attention},
    booktitle={International Conference on Multimedia and Expo},
    year={2023},
}
```
---
## Credit
Some parts of this code (e.g., data_loader) are based on [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) repository.

