# Quick Start (Video Captioning)

The following guideline shows how to train and evaluate on [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities).

## Training

### Train baseline model

Use the following command to start the training of the baseline model.

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/baseline.yml'
```

### Train with Cyclical regimen

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --path_opt 'cfgs/cyclical.yml'
```

## Evaluation


## Test server submission

Change the `val_split` and `densecap_references` accordingly as follows. Check the instruction provided in [GVD GitHub repo](https://github.com/facebookresearch/grounded-video-description#inference-and-testing) for more infomation regarding test (or hidden test) server submission.

```yml
val_split: validation
densecap_references: ["anet_entities_val_1.json", "anet_entities_val_2.json"]

# use the following two lines for test set submission instead
val_split: testing
densecap_references: ["anet_entities_test_1.json", "anet_entities_test_2.json"]
```

You might want to change `*testing*.json` to `*hidden_test*.json` for submission of hidden test set.
