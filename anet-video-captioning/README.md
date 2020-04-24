# Quick Start (Video Captioning)

The following guideline shows how to train and evaluate on [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities).

## Training

### Train baseline model

Use the following command to start the training of our baseline model.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --path_opt 'cfgs/baseline.yml'

CUDA_VISIBLE_DEVICES=0,1 python main.py
```

## Evaluation
