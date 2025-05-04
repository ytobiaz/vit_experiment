![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# Design and Implementation of an Efficient Vision Transformer via Knowledge Distillation and Pruning

This repository contains the code of a bachelor's thesis examining the feasibility of combining knowledge distillation and pruning into a joint approach to achieve an efficient Vision Transformer model.

---

## Requirements

The code was developed and tested in a virtual environment using **Python 3.8**, with **PyTorch 2.1.0** and **torchvision 0.16.0**.

In addition to PyTorch, the following libraries are required:

- [timm 0.6.12](https://github.com/rwightman/pytorch-image-models)  
- [einops](https://github.com/arogozhnikov/einops)  
- [tensorboardX](https://github.com/lanpa/tensorboardX)

To install all dependencies at once:

```bash
pip install -r requirements.txt
```

---


### Data Preparation

Please download the ImageNet dataset from [`http://www.image-net.org/`](http://www.image-net.org/). The training and validation images need to be organized into subfolders for each class.

The train set and validation set should be saved as the `*.tar` archives:

```
ImageNet/
├── train.tar
└── val.tar
```

The code also supports storing images as individual files:

```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
```


---

## Acknowledgement

This repository builds on the [timm](https://github.com/huggingface/pytorch-image-models) library. I thank [Ross Wightman](https://rwightman.com/) for creating and maintaining this high-quality resource.

Additionally, this work is heavily based on [NViT](https://github.com/NVlabs/NViT) and integrates components from [CSKD](https://github.com/Zzzzz1/CSKD). I made my own modifications to ensure compatibility and seamless integration. I thank the respective authors for their contributions and for sharing their codebases.
