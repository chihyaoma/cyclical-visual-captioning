#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script to download all the necessary data files and place under the data directory
# Written by Luowei Zhou on 05/01/2019
# Revised by Chih-Yao Ma on 04/13/2020


DATA_ROOT='data'

mkdir -p $DATA_ROOT/anet save results log

# annotation files
# 3 MB
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz
# 4 MB
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz
# 25 MB
wget -P tools/coco-caption/annotations https://github.com/jiasenlu/coco-caption/raw/master/annotations/caption_flickr30k.json
# 4 MB
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz

# feature files
# 32 MB
wget -P $DATA_ROOT/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz
# 39 GB (83 GB after untar)
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
# 2.9 GB
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5
# 169 GB (419 GB after untar)
wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz

# Stanford CoreNLP 3.9.1
# 372 MB
wget -P tools/ http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

# pre-trained models
# 1.4 GB
wget -P save/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/pre-trained-models.tar.gz

# uncompress
cd $DATA_ROOT
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
cd anet
for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
cd ../../tools
for file in *.zip; do unzip "${file}" && rm "${file}"; done

# cd coco-caption
# ./get_stanford_models.sh
# cd ../../save
# for file in *.tar.gz; do tar -zxvf "${file}" && rm "${file}"; done
# mv pre-trained-models/* . && rm -r pre-trained-models
