#!/bin/bash

data_dir=/home/hqface/HQ-Face-Video-Dataset
video_dir=${data_dir}/video
frame_dir=${data_dir}/tmp/frame_2fps
subtitle_dir=$
tmp_output_dir=${data_dir}/tmp

echo "Add label."
python add_label.py --input_dir=${data_dir}/DATA/week1/output_hp/bbox_hp.json --label_dir=${data_dir}/DATA/week1/tmp/week1.xlsx --output_dir=${data_dir}/DATA/week1/output_hp/new_bbox_hp.json
echo "Finished"