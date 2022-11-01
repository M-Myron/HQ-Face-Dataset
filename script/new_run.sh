#!/bin/bash

data_dir=/home/hqface/HQ-Face-Video-Dataset/DATA/week2
video_dir=${data_dir}/video
frame_dir=${data_dir}/tmp/frame_2fps
tmp_output_dir=${data_dir}/tmp
stage=0

mkdir -p ${tmp_output_dir} || exit 1
mkdir -p ${frame_dir} || exit 1
mkdir -p ${tmp_output_dir}/scene || exit 1

if [ $stage -le 0 ]; then
    echo "Start frame extract"
    sh extract_frame.sh
    echo "Finish frame extract"
fi
    
if [ $stage -le 1 ]; then    
#     sh scene_detect.sh &
#     sh face_det.sh &
#     wait
#     sh face_det.sh
    echo "Preparation is finished"
    
    echo "Stage 0: Merge the time stamp."
    # input face time, output time stamp.
    python merge.py --face_time_path=${tmp_output_dir}/face_time.txt --output_dir=${tmp_output_dir}/time_stamp.txt --scene_detect_dir=${tmp_output_dir}/scene || exit 1
    echo "Finished"
#     exit 0
fi

if [ $stage -le 2 ]; then
    echo "Stage 2: Seperate into clips with subtitle time stamps."
    # input 2fps_timestamp, output clip_timestamp, save time stamps (json) and subtitle (text)
    python clip_with_subtitle.py --timestamp=${tmp_output_dir}/time_stamp.txt --subtitle_dir=${data_dir}/info/subtitle --output_dir=${tmp_output_dir}
    echo "Stage 2 finished."
#     exit 0
fi

if [ $stage -le 3 ]; then
    echo "Stage 3: Human parsing, get frame bboxes."
    python human_parsing_bbox.py --clip_info=${tmp_output_dir}/clip_info.json --frame_path=${frame_dir} --output_dir=${tmp_output_dir}
    echo "Stage 3 finished."
#     exit 0 
fi

if [ $stage -le 4 ]; then
    echo "Stage 4: Get clip bboxes and video info"
    python get_bbox_hp.py --hp_bbox=${tmp_output_dir}/human_parsing_bbox.json --video_path_list=${data_dir}/index/uid2path --output_dir=${data_dir}/output_hp
    echo "Stage 4 finished."
#     exit 0
fi

if [ $stage -le 5 ]; then
    echo "Stage 5: add new albels"
    python add_label.py --input_dir=${data_dir}/DATA/week1/output_hp/bbox_hp.json --label_dir=${data_dir}/DATA/week1/tmp/week1.xlsx --output_dir=${data_dir}/DATA/week1/output_hp/new_bbox_hp.json
    echo "Stage 5 finished."
#     exit 0
fi    