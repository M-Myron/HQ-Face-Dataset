import ffmpeg
import numpy as np
import json
import os
import argparse
import itertools
# import seaborn as sns
# import matplotlib.pyplot as plt

def get_arm_label(labels):
    len_list = []
    for k, v in itertools.groupby(labels):
        if k == 1:
            len_list.append(len(list(v)))
    if len_list:
        if max(len_list) >= 2:
            return 1
        else:
            return 0
    else:
        return 0

def get_bbox_hp(hp_bbox, video_path_list, output_dir):
    with open(hp_bbox) as f:
        dict = json.load(f)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    video_path_dict = {}
    with open(video_path_list,'r') as f:
        lines = f.readlines()
        for line in lines:
            value = line.strip().split()
            video_path_dict[value[0]] = value[1]

    bbox_list = []
    video_num = 0
    video_duration = 0
    clip_num = 0
    clip_duration = 0
    no_clip_uid = []
    # face_w, face_h = [], []
    # upper_half_w, upper_half_h = [], []
    # upper_body_w, upper_body_h = [], []
    # upper_full_w, upper_full_h = [], []
    for video in dict:
        uid = video['uid']
        video_path = video_path_dict[uid]
        if not os.path.isfile(video_path):
            print("video uid={} does not exist.".format(uid))
            continue
        print(video['uid'], len(video['clips_info']))
        if not video['clips_info']:
            no_clip_uid.append(uid)
            print("uid={}, no clip!".format(uid))
            continue
#         if os.path.isfile(os.path.join(video_dir, "{}.mp4".format(uid))):
#             video_path = os.path.join(video_dir, "{}.mp4".format(uid))
#         elif os.path.isfile(os.path.join(video_dir, "{}.mkv".format(uid))):
#             video_path = os.path.join(video_dir, "{}.mkv".format(uid))
#         elif os.path.isfile(os.path.join(video_dir, "{}.webm".format(uid))):
#             video_path = os.path.join(video_dir, "{}.webm".format(uid))
#         else:
#             print("uid={} does not exist.")
#             continue
        # get fps
        probe = ffmpeg.probe(video_path)
        format = probe['format']
        video_duration += float(format['duration'])
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found:' + video_path)
            continue
        video_num += 1
        fps = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
        bbox_dict = {'uid': uid, 'video_width': video['video_width'], 'video_height': video['video_height'],
                     'video_fps': round(fps),
                     'clip_info': []}
        for id, clip in enumerate(video['clips_info']):
            if not clip['bbox']:
                continue
            clip_num += 1
            start_time = clip['start_time']
            end_time = clip['end_time']
            clip_duration += end_time - start_time
            if len(clip['bbox']) > 1:
                face = []
                upper_half = []
                upper_body = []
                upper_full = []
                face_arm = []
                upper_half_arm = []
                upper_body_arm = []
                upper_full_arm = []
                for frame in clip['bbox']:
                    face.append(frame['face_bbox'])
                    upper_half.append(frame['upper_half_bbox'])
                    upper_body.append(frame['upper_body_bbox'])
                    upper_full.append(frame['upper_full_bbox'])
                    face_arm.append(frame['arms_label']['face_bbox'])
                    upper_half_arm.append(frame['arms_label']['upper_half_bbox'])
                    upper_body_arm.append(frame['arms_label']['upper_body_bbox'])
                    upper_full_arm.append(frame['arms_label']['upper_full_bbox'])
                face = np.array(face)
                upper_half = np.array(upper_half)
                upper_body = np.array(upper_body)
                upper_full = np.array(upper_full)
                face_arm = np.array(face_arm)
                upper_half_arm = np.array(upper_half_arm)
                upper_body_arm = np.array(upper_body_arm)
                upper_full_arm = np.array(upper_full_arm)
                face_bbox = []
                upper_half_bbox = []
                upper_body_bbox = []
                upper_full_bbox = []
                err_frame_face = []
                err_frame_half = []
                err_frame_body = []
                for i in range(4):
                    face_i = face[:, i]
                    while 1:
                        mean = np.nanmean(face_i)
                        std = np.nanstd(face_i)
                        face_i[face_i > (mean + 3 * std)] = np.nan
                        face_i[face_i < (mean - 3 * std)] = np.nan
                        if mean == np.nanmean(face_i) and std == np.nanstd(face_i):
                            err_frame_face.extend(np.argwhere(np.isnan(face_i)))
                            break
                    half_i = upper_half[:, i]
                    while 1:
                        mean = np.nanmean(half_i)
                        std = np.nanstd(half_i)
                        half_i[half_i > (mean + 3 * std)] = np.nan
                        half_i[half_i < (mean - 3 * std)] = np.nan
                        if mean == np.nanmean(half_i) and std == np.nanstd(half_i):
                            err_frame_half.extend(np.argwhere(np.isnan(half_i)))
                            break
                    body_i = upper_body[:, i]
                    while 1:
                        mean = np.nanmean(body_i)
                        std = np.nanstd(body_i)
                        body_i[body_i > (mean + 3 * std)] = np.nan
                        body_i[body_i < (mean - 3 * std)] = np.nan
                        if mean == np.nanmean(body_i) and std == np.nanstd(body_i):
                            err_frame_body.extend(np.argwhere(np.isnan(body_i)))
                            break
                    if i <= 1:
                        face_bbox.append(np.nanmin(face_i))
                        upper_half_bbox.append(np.nanmean(half_i))
                        upper_body_bbox.append(np.nanmean(body_i))
                        upper_full_bbox.append(np.min(upper_full[:, i]))
                    else:
                        face_bbox.append(np.nanmax(face_i))
                        upper_half_bbox.append(np.nanmean(half_i))
                        upper_body_bbox.append(np.nanmean(body_i))
                        upper_full_bbox.append(np.max(upper_full[:, i]))
                if err_frame_face:
                    err_frame_face = np.unique(np.array(err_frame_face))
                    face_arm[err_frame_face] = 0
                if err_frame_half:
                    err_frame_half = np.unique(np.array(err_frame_half))
                    upper_half_arm[err_frame_half] = 0
                if err_frame_body:
                    err_frame_body = np.unique(np.array(err_frame_body))
                    upper_body_arm[err_frame_body] = 0
                face_arm_label = get_arm_label(face_arm)
                upper_half_arm_label = get_arm_label(upper_half_arm)
                upper_body_arm_label = get_arm_label(upper_body_arm)
                upper_full_arm_label = get_arm_label(upper_full_arm)
            else:
                frame = clip['bbox'][0]
                face_bbox = frame['face_bbox']
                upper_half_bbox = frame['upper_half_bbox']
                upper_body_bbox = frame['upper_body_bbox']
                upper_full_bbox = frame['upper_full_bbox']
                face_arm_label = frame['arms_label']['face_bbox']
                upper_half_arm_label = frame['arms_label']['upper_half_bbox']
                upper_body_arm_label = frame['arms_label']['upper_body_bbox']
                upper_full_arm_label = frame['arms_label']['upper_full_bbox']
            clip_dict = {'start_time': start_time, 'end_time': end_time, 'text': clip['text'],
                         'bbox': {'face_bbox': face_bbox,
                                  'upper_half_bbox': upper_half_bbox,
                                  'upper_body_bbox': upper_body_bbox,
                                  'upper_full_bbox': upper_full_bbox},
                         'arms_label': {'face_bbox': face_arm_label,
                                        'upper_half_bbox': upper_half_arm_label,
                                        'upper_body_bbox': upper_body_arm_label,
                                        'upper_full_bbox': upper_full_arm_label}}
            bbox_dict['clip_info'].append(clip_dict)
            # face_w.append(face_bbox[2] - face_bbox[0])
            # face_h.append(face_bbox[3] - face_bbox[1])
            # upper_half_w.append(upper_half_bbox[2] - upper_half_bbox[0])
            # upper_half_h.append(upper_half_bbox[3] - upper_half_bbox[1])
            # upper_body_w.append(upper_body_bbox[2] - upper_body_bbox[0])
            # upper_body_h.append(upper_body_bbox[3] - upper_body_bbox[1])
            # upper_full_w.append(upper_full_bbox[2] - upper_full_bbox[0])
            # upper_full_h.append(upper_full_bbox[3] - upper_full_bbox[1])
        bbox_list.append(bbox_dict)

    stat = "no clip uid:{}\n".format(
        no_clip_uid) + "valid video num:{}, valid video duration:{:.02f}min\nclip num:{}, clip duration:{:.02f}min".format(
        video_num, video_duration / 60, clip_num, clip_duration / 60)
    with open(os.path.join(output_dir, 'stat_hp.txt'), 'w') as f:
        print(stat, file=f)
    print(stat)
    with open(os.path.join(output_dir, 'bbox_hp.json'), 'w') as f:
        json.dump(bbox_list, f)

    # face_w = np.array(face_w)
    # sns.displot(face_w)
    # plt.title("face_bbox avg_w={:.02f}".format(np.mean(face_w)))
    # plt.tight_layout()
    # face_h = np.array(face_h)
    # sns.displot(face_h)
    # plt.title("face_bbox avg_h={:.02f}".format(np.mean(face_h)))
    # plt.tight_layout()
    # upper_half_w = np.array(upper_half_w)
    # sns.displot(upper_half_w)
    # plt.title("upper_half_bbox avg_w={:.02f}".format(np.mean(upper_half_w)))
    # plt.tight_layout()
    # upper_half_h = np.array(upper_half_h)
    # sns.displot(upper_half_h)
    # plt.title("upper_half_bbox avg_h={:.02f}".format(np.mean(upper_half_h)))
    # plt.tight_layout()
    # upper_body_w = np.array(upper_body_w)
    # sns.displot(upper_body_w)
    # plt.title("upper_body_bbox avg_w={:.02f}".format(np.mean(upper_body_w)))
    # plt.tight_layout()
    # upper_body_h = np.array(upper_body_h)
    # sns.displot(upper_body_h)
    # plt.title("upper_body_bbox avg_h={:.02f}".format(np.mean(upper_body_h)))
    # plt.tight_layout()
    # upper_full_w = np.array(upper_full_w)
    # sns.displot(upper_full_w)
    # plt.title("upper_full_bbox avg_w={:.02f}".format(np.mean(upper_full_w)))
    # plt.tight_layout()
    # upper_full_h = np.array(upper_full_h)
    # sns.displot(upper_full_h)
    # plt.title("upper_full_bbox avg_h={:.02f}".format(np.mean(upper_full_h)))
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_bbox', type=str)
    parser.add_argument('--video_path_list', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    get_bbox_hp(args.hp_bbox, args.video_path_list, args.output_dir)
