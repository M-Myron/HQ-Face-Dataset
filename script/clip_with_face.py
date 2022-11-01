import argparse
import re
import os
import pysrt
import glob
import time

from joblib import Parallel, delayed
from tqdm import tqdm
import json
# from json_utils import load_annotation_list, save_jsonl
# from configs.preprocess_configs import DOWNLOAD_ROOT, SUBTITLE_ROOT


def convert_sub_time_to_seconds(sub_time):
    """sub_time is a SubRipTime object defined by pysrt"""
    return 60 * sub_time.minutes + sub_time.seconds + 0.001 * sub_time.milliseconds


def clean_single_sub_sentence(sub_sentence):
    """sub_sentence: str, """
    sub_sentence = sub_sentence.replace("\n", " ")
    sub_sentence = sub_sentence.replace("(", " ")
    sub_sentence = sub_sentence.replace(")", " ")
    sub_sentence = sub_sentence.replace(":", " : ")
    sub_sentence = re.sub(r"\s{2,}", " ", sub_sentence)  # 这里是正则表达式，单空格替换双空格 https://www.runoob.com/python/python-reg-expressions.html
    return sub_sentence


def split_multi_lines(cur_sub):
    """
    return: subtitle list that split all multi-line items
    """
    start_time = []
    end_time = []
    duration = cur_sub.duration
    text_length = len(cur_sub.text)
    text = cur_sub.text.split('\n')

    start_time.append(cur_sub.start)
    for i in range(len(text) - 1):
        middle_time = start_time[i].__add__(duration.__mul__(len(text[i]) / text_length))
        start_time.append(middle_time)
        end_time.append(middle_time)
    end_time.append(cur_sub.end)

    splited_item = dict(
        text_list=text,
        start_time_list=start_time,
        end_time_list=end_time
    )
    return splited_item


def preprocess_subtitles_single_video(video_path):
    """
    return: A python dict, the keys are the video names, the entries are lists,
            each contains all the text from a .srt file
    sub_times are the start time of the sentences.
    """
    video_id = os.path.basename(video_path).split('.')[0]
    subs = pysrt.open(video_path, encoding="iso-8859-1")  # 打开字幕.srt文件到subs
    if len(subs) == 0:
        subs = pysrt.open(video_path)

    # 转成单行
    sub_single_line = []
    for cur_sub in subs:
        # 对读取到subs中的每一个srt item进行处理
        text = cur_sub.text
        if text == '':
            continue

        if '\n' in text:
            splited_item = split_multi_lines(cur_sub)
            text_list = splited_item['text_list']
            start_time_list = splited_item['start_time_list']
            end_time_list = splited_item['end_time_list']
            for sentence_index in range(len(text_list)):
                sub_single_line.append(dict(
                    text=text_list[sentence_index],
                    start=start_time_list[sentence_index],
                    end=end_time_list[sentence_index]
                ))
        else:
            sub_single_line.append(dict(
                text=cur_sub.text,
                start=cur_sub.start,
                end=cur_sub.end
            ))

    # 去重
    sub_data = []
    prev = sub_single_line[0]
    for sentence_index in range(1, len(sub_single_line)):
        cur_sub = sub_single_line[sentence_index]
        if cur_sub['text'] != prev['text'] and "<c>" not in cur_sub['text']:
            sub_data.append(dict(
                text=clean_single_sub_sentence(prev['text']),
                start=convert_sub_time_to_seconds(prev['start']),
                end=convert_sub_time_to_seconds(cur_sub['start'])
            ))
            prev = cur_sub

        if sentence_index == len(sub_single_line) - 1:
            sub_data.append(dict(
                text=clean_single_sub_sentence(prev['text']),
                start=convert_sub_time_to_seconds(prev['start']),
                end=convert_sub_time_to_seconds(cur_sub['end'])
            ))
    return sub_data

    # 划分入segment，并在segment分界点拆开
    # split_point_list = [segment_list[0][0]] + [segment_list[i][1] for i in range(len(segment_list))]
    # split_point_index = 0
    # sentence_index = 0
    # while sentence_index < len(sub_data) and split_point_index < len(split_point_list):
    #     split_point = split_point_list[split_point_index]
    #     if sub_data[sentence_index]['start'] < split_point < sub_data[sentence_index]['end']:
    #         text = sub_data[sentence_index]['text']
    #         start = sub_data[sentence_index]['start']
    #         end = sub_data[sentence_index]['end']
    #
    #         # split the sentence into two
    #         n_characters = len(text) * ((split_point - start) / (end - start))
    #         text = text.split(' ')
    #         text_before = ''
    #         text_after = ''
    #         for word in text:
    #             if len(text_before) + len(word) <= n_characters:
    #                 text_before = text_before + word + ' '
    #             else:
    #                 text_after = text_after + word + ' '
    #
    #         # put the split item back to list
    #         sub_data[sentence_index]['end'] = split_point
    #         sub_data[sentence_index]['text'] = text_before[:-1]
    #         sub_data.insert(sentence_index + 1, dict(
    #             text=text_after[:-1],
    #             start=split_point,
    #             end=end
    #         ))
    #         split_point_index += 1
    #         sentence_index += 1
    #     elif sub_data[sentence_index]['start'] >= split_point:
    #         split_point_index += 1
    #     else:
    #         sentence_index += 1

    # attach to every segment
    # seg_sub_data = [[] for i in range(len(segment_list))]
    # for sub in sub_data:
    #     if not sub['text'] == '':
    #         for seg_index in range(len(segment_list)):
    #             seg_start = segment_list[seg_index][0]
    #             seg_end = segment_list[seg_index][1]
    #             sentence_start = sub['start']
    #             sentence_end = sub['end']
    #             if seg_start <= sentence_start and sentence_end <= seg_end:
    #                 seg_sub_data[seg_index].append(sub)

    # write to json
    # srt_data_list = []
    # for seg_index in range(len(segment_list)):
    #     srt_data = dict(
    #         seg_id=video_id + '_' + str(seg_index),
    #         sub=seg_sub_data[seg_index]
    #     )
    #     srt_data_list.append(srt_data)
    # save_jsonl(srt_data_list, os.path.join(save_path, video_id + '.jsonl'))


# def preprocess_subtitles(segment, srt_dir=DOWNLOAD_ROOT, save_path=SUBTITLE_ROOT):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     print("Start preprocessing srt files from %s ..." % srt_dir)
#     srt_paths = glob.glob(os.path.join(srt_dir, "*.srt"))
#     for i in range(len(srt_paths)):
#         srt = srt_paths[i]
#         if os.path.exists(save_path + '/' + srt[-19:-7] + '.jsonl'):
#             srt_paths.remove(srt)
#     Parallel(n_jobs=32)(delayed(preprocess_subtitles_single_video)
#                         (srt, save_path, segment[os.path.basename(srt).split('.')[0]])
#                         for srt in tqdm(srt_paths, desc="Loop over subtitle files"))


# if __name__ == '__main__':
#     # Get segment info
#     # seg_info = dict()
#     # annotation_list = load_annotation_list()
#     # for anno in annotation_list:
#     #     seg_info[anno[0]['videoID']] = anno[1]['segInfo']

#     sub_path = "./DATA/info/subtitle/subtitle_manual/9l3f1KrMQeo.en-US.vtt"
#     preprocess_subtitles_single_video(sub_path)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    t_0 = time.time()
    
    time_stp_list = [item.split() for item in open(args.timestamp)]
    uid2stp = {}
    
    for line in time_stp_list:
        uid = line[0]
        start_t = float(line[1])/30
        end_t = float(line[2])/30
        
        if uid in uid2stp.keys():
            uid2stp[uid].append([start_t, end_t])
        else:
            uid2stp[uid] = [[start_t, end_t]]
        
    # sub_file_list = os.listdir(args.subtitle_dir)
    # uid2sub = {}
    
    # for f in sub_file_list:
    #     uid = f.split(".")[0]
    #     uid2sub[uid] = preprocess_subtitles_single_video(os.path.join(args.subtitle_dir, f))
    
    uid2clip = {}
    for uid in uid2stp.keys():
        face_slots = uid2stp[uid]
        uid2clip[uid] = []
        for slot in face_slots:
            if slot[1]-slot[0] > 10:
                uid2clip[uid].append({"text":'', "start":slot[0], "end":slot[1]})

#     for uid in uid2sub:
# #         print(uid)
#         if uid in uid2stp.keys():
#             face_slot = uid2stp[uid]
#         #     print("face slot", face_slot)
#         #     print("sub slot", uid2sub[uid])
#             cur_slot = 0
#             clip_start = face_slot[0][0]
#             clip_text = ''
#             min_len = 5
#             uid2clip[uid] = []
#             flag = 0

#             for n_sent, sent in enumerate(uid2sub[uid]):
#                 if cur_slot == len(face_slot):
#                     break
#                 start_t = float(sent["start"])
#                 end_t = float(sent["end"])
#                 text = sent["text"]
#                 if flag == 0:
#                     if start_t >= face_slot[cur_slot][1]:
#                         cur_slot += 1
#                         continue
#         #             print("Current slot:",cur_slot, face_slot[cur_slot][0], face_slot[cur_slot][1])
#         #             print(sent)
#                     if start_t >= face_slot[cur_slot][0] and end_t <= face_slot[cur_slot][1]:
#         #                 print("sent in slot")
#                         clip_start = start_t
#                         clip_end = end_t
#                         clip_text = text
#                         if clip_end - clip_start >= min_len:
#                             uid2clip[uid].append({"text":clip_text, "start":clip_start, "end":clip_end})
#     #                         print(uid2clip[uid][-1])
#         #                     print("end record")
#                             clip_text = ""
#                         else:
#                             flag = 1
#         #                     print("start record")

#                 else:
#                     if start_t >= face_slot[cur_slot][0] and end_t <= face_slot[cur_slot][1]:
#                         clip_end = end_t
#                         clip_text += " "+text
#                         if clip_end - clip_start >= min_len:
#                             uid2clip[uid].append({"text":clip_text, "start":clip_start, "end":clip_end})
#                             clip_text = ""
#                             flag = 0
#         #                     print(uid2clip[uid][-1])
#         #                     print("end record")
#                     else:
#                         flag = 0
#         #                 print("not in slot, next clip")

    output = json.dumps(uid2clip, indent=4, separators=(',', ': '))
    f = open(os.path.join(args.output_dir, "scene_clip_info.json"), "w")
    f.write(output)
    f.close()
    
    clip_num = 0
    clip_dur = 0
    vid_num = len(uid2clip)
    for item in uid2clip:
        clip_num += len(uid2clip[item])
        for sent in uid2clip[item]:
            clip_dur += float(sent["end"])-float(sent["start"])
        
    print("Get", clip_num, "clips from", vid_num, "videos.")
    print("Total duration of all clips is", int(clip_dur/60), "minutes.")
    t_1 = time.time()
    print("Cost time:", t_1-t_0)
    
    