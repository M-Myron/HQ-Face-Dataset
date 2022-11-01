#!/bin/bash

data_dir=$1
video_dir=$2
frame_dir=$3
tmp_output_dir=$4

echo "Face detection and face recognition."
python -u face_det.py --data_dir=${data_dir} --face_path=${data_dir}/info/face --frame_path=${frame_dir} --output_dir=${tmp_output_dir}/face_time.txt --landmark_dir=${tmp_output_dir}/landmark.csv
echo "Finished"