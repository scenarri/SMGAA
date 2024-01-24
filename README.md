## SMGAA
Code for the paper "[Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense](https://ieeexplore.ieee.org/abstract/document/9915465)".  It will be available soon.

## Preparation
Please download [model weights](https://pan.baidu.com/s/1HKW6VkEybYB0M3N84IInjg?pwd=5631&_at_=1706016266118#list/path=%2F) and our [test set](https://pan.baidu.com/s/1p8sUmnCXwnRdt7vMFomWJg?pwd=5631), and arrange them to './models/' and './dataset/' respectively. Please modify your torchvision.datasets.folder with the following code to automatically load the .pt file:

```
import torch

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".pt")

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        if path[-3:] == '.pt':
            return torch.load(path)
        else:
            return pil_loader(path)
```

## Evaluation
Run the command below to evaluate SMGAA attack, where '--model' assigns the victim model, '--nb_asc' set the number of adversarial scatterers, '--popbatch' is the population in search, '--maxiter' is the number of search iterations, and '--SAVE' will assure the results be saved at './Record/'.  Assigning '--restart' leads to a total population of popbatch\*restart that along with some models may exceed GPU memory limitation.  That is, you can decompose an interested population number to popbatch\*restart to satisfy hardware conditions.   This may consume more time and lead to slightly different results due to seed setting.
```
python SMGAA.py --model alex/vgg/res/dense/mobile/aconv/shuffle/squeeze --nb_asc 2 --popbatch 100 --maxiter 90 --restart 1
```

## Citation
If you find our paper and this repository useful, please consider citing our work.
```bibtex
@ARTICLE{pengsmgaa22,
  author={Peng, Bowen and Peng, Bo and Zhou, Jie and Xie, Jianyue and Liu, Li},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense}, 
  year={2022},
  volume={60},
  pages={1-17},
  doi={10.1109/TGRS.2022.3213305}}
```
