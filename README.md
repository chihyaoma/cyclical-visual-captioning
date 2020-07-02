## Learning to Generate Grounded Visual Captions without Localization Supervision
<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch implementation of our paper:

**Learning to Generate Grounded Visual Captions without Localization Supervision**<br>
[__***Chih-Yao Ma***__](https://chihyaoma.github.io/), [Yannis Kalantidis](https://www.skamalas.com/), [Ghassan AlRegib](https://ghassanalregib.info/), [Peter Vajda](https://sites.google.com/site/vajdap), 
[Marcus Rohrbach](https://rohrbach.vision/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/)<br>
European Conference on Computer Vision (ECCV), 2020 <br>

[[arXiv](https://arxiv.org/abs/1906.00283)] [[GitHub](https://github.com/chihyaoma/cyclical-visual-captioning)] [[Project](https://chihyaoma.github.io/project/2019/06/03/cyclical.html)]

<p align="center">
<img src="teaser/concept.png" width="100%">
</p>

## How to start

Clone the repo recursively:

```shell
git clone --recursive git@github.com:chihyaoma/cyclical-visual-captioning.git
```

If you didn't clone with the --recursive flag, then you'll need to manually clone the pybind submodule from the top-level directory:

```shell
git submodule update --init --recursive
```

## Installation

The proposed cyclical method can be applied directly to image and video captioning tasks.

Currently, installation guide and our code for video captioning on the ActivityNet-Entities dataset are provided in [anet-video-captioning](anet-video-captioning).

## Acknowledgments

Chih-Yao Ma and Zsolt Kira were partly supported by DARPA’s Lifelong Learning Machines (L2M) program, under Cooperative Agreement HR0011-18-2-0019, as part of their affiliation with Georgia Tech.
We thank Chia-Jung Hsu for her valuable and artistic helps on the figures.

## Citation

If you find this repository useful, please cite our paper:

```shell
@inproceedings{ma2020learning,
    title={Learning to Generate Grounded Image Captions without Localization Supervision},
    author={Ma, Chih-Yao and Kalantidis, Yannis and AlRegib, Ghassan and Vajda, Peter and Rohrbach, Marcus and Kira, Zsolt},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020},
    url={https://arxiv.org/abs/1906.00283},
}
```
