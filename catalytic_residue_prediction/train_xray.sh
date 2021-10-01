#!/bin/bash

model_name = "train_xray"

python deepfri/train_DeepFRI.py -gc GraphConv -pd 990 -ont ec -lm data/lstm_lm_tf.hdf5 --model_name $model_name --train_tfrecord_fn data/$model_name/train --valid_tfrecord_fn data/$model_name/val --annot_fn data/annot_files/train_val_annot.tsv --epochs 100 --load_and_continue
