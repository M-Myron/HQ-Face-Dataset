import ffmpeg
import json
import os
import argparse


def crop_clip(bbox_hp, video_dir, output_dir):
    with open(bbox_hp) as f:
        dict = json.load(f)
    if not os.path.exists(output_dir):
            print('----------')
            os.mkdir(output_dir)
    for vid, video in enumerate(dict):
        uid = video['uid']
#         print("Processing",uid)
        if not video['clip_info']:
            print("uid={}, no clip!".format(uid))
            continue
        if os.path.isfile(os.path.join(video_dir, "{}.mp4".format(uid))):
            video_path = os.path.join(video_dir, "{}.mp4".format(uid))
        elif os.path.isfile(os.path.join(video_dir, "{}.mkv".format(uid))):
            video_path = os.path.join(video_dir, "{}.mkv".format(uid))
        elif os.path.isfile(os.path.join(video_dir, "{}.webm".format(uid))):
            video_path = os.path.join(video_dir, "{}.webm".format(uid))
        else:
            print("uid={} does not exist.")
            continue
        print("processing uid={}, {}/{}".format(uid, vid, len(dict)))
        # if uid != '11jiBIjhKZ0':
        #     continue
        # video_path = "./download/11jiBIjhKZ0.mp4"
        uid_dir = os.path.join(output_dir, '{}'.format(uid))
        if not os.path.exists(uid_dir):
            os.mkdir(uid_dir)
        
        face_bbox = video['clip_info'][0]['bbox']['face_bbox']
        for id, clip in enumerate(video['clip_info']):
            face_bbox[0] = min(face_bbox[0], clip['bbox']['face_bbox'][0])
            face_bbox[1] = min(face_bbox[1], clip['bbox']['face_bbox'][1])
            face_bbox[2] = max(face_bbox[2], clip['bbox']['face_bbox'][2])
            face_bbox[3] = max(face_bbox[3], clip['bbox']['face_bbox'][3])
        face_x = face_bbox[0]
        face_y = face_bbox[1]
        face_w = face_bbox[2] - face_bbox[0]
        face_h = face_bbox[3] - face_bbox[1]

        vid = ffmpeg.input(video_path)
        video_clip = vid.video
        audio = vid.audio

        face_dir = os.path.join(uid_dir, "face")
        if not os.path.exists(face_dir):
            os.mkdir(face_dir)
        face_clip = ffmpeg.crop(video_clip, face_x, face_y, face_w, face_h)
#             face_clip = ffmpeg.output(face_clip, os.path.join(face_dir, '%05d.jpg'))
        face_clip = ffmpeg.output(face_clip, audio, os.path.join(face_dir, 'vid.mp4'))
        ffmpeg.run(face_clip, overwrite_output=True, quiet=True)




#             if not clip['bbox']:
#                 continue
#             clip_dir = os.path.join(uid_dir, '{}'.format(id))
#             if not os.path.exists(clip_dir):
#                 os.mkdir(clip_dir)
#             start_time = clip['start_time']
#             end_time = clip['end_time']
#             # vid = ffmpeg.input(video_path, ss=start_time, to=end_time)
#             vid = ffmpeg.input(video_path)
#             video_clip = vid.video
#             audio = vid.audio

#             face_dir = os.path.join(clip_dir, "face")
#             if not os.path.exists(face_dir):
#                 os.mkdir(face_dir)
#             face_x = clip['bbox']['face_bbox'][0]
#             face_y = clip['bbox']['face_bbox'][1]
#             face_w = clip['bbox']['face_bbox'][2] - clip['bbox']['face_bbox'][0]
#             face_h = clip['bbox']['face_bbox'][3] - clip['bbox']['face_bbox'][1]
# #             print(video_clip, face_x, face_y, face_w, face_h)
#             face_clip = ffmpeg.crop(video_clip, face_x, face_y, face_w, face_h)
# #             face_clip = ffmpeg.output(face_clip, os.path.join(face_dir, '%05d.jpg'))
#             face_clip = ffmpeg.output(face_clip, audio, os.path.join(face_dir, 'vid.mp4'))
#             ffmpeg.run(face_clip, overwrite_output=True, quiet=True)

#             upper_half_dir = os.path.join(clip_dir, "upper_half")
#             if not os.path.exists(upper_half_dir):
#                 os.mkdir(upper_half_dir)
#             upper_half_x = clip['bbox']['upper_half_bbox'][0]
#             upper_half_y = clip['bbox']['upper_half_bbox'][1]
#             upper_half_w = clip['bbox']['upper_half_bbox'][2] - clip['bbox']['upper_half_bbox'][0]
#             upper_half_h = clip['bbox']['upper_half_bbox'][3] - clip['bbox']['upper_half_bbox'][1]
# #             print(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
#             upper_half_clip = ffmpeg.crop(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
# #             upper_half_clip = ffmpeg.output(upper_half_clip, os.path.join(upper_half_dir, '%05d.jpg'))
#             upper_half_clip = ffmpeg.output(upper_half_clip, audio, os.path.join(upper_half_dir, 'vid.mp4'))
#             ffmpeg.run(upper_half_clip, overwrite_output=True, quiet=True)
        
#             upper_half_dir = os.path.join(clip_dir, "upper_body")
#             if not os.path.exists(upper_half_dir):
#                 os.mkdir(upper_half_dir)
#             upper_half_x = clip['bbox']['upper_body_bbox'][0]
#             upper_half_y = clip['bbox']['upper_body_bbox'][1]
#             upper_half_w = clip['bbox']['upper_body_bbox'][2] - clip['bbox']['upper_body_bbox'][0]
#             upper_half_h = clip['bbox']['upper_body_bbox'][3] - clip['bbox']['upper_body_bbox'][1]
# #             print(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
#             upper_half_clip = ffmpeg.crop(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
# #             upper_half_clip = ffmpeg.output(upper_half_clip, os.path.join(upper_half_dir, '%05d.jpg'))
#             upper_half_clip = ffmpeg.output(upper_half_clip, audio, os.path.join(upper_half_dir, 'vid.mp4'))
#             ffmpeg.run(upper_half_clip, overwrite_output=True, quiet=True)
        
#             upper_half_dir = os.path.join(clip_dir, "upper_full")
#             if not os.path.exists(upper_half_dir):
#                 os.mkdir(upper_half_dir)
#             upper_half_x = clip['bbox']['upper_full_bbox'][0]
#             upper_half_y = clip['bbox']['upper_full_bbox'][1]
#             upper_half_w = clip['bbox']['upper_full_bbox'][2] - clip['bbox']['upper_full_bbox'][0]
#             upper_half_h = clip['bbox']['upper_full_bbox'][3] - clip['bbox']['upper_full_bbox'][1]
# #             print(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
#             upper_half_clip = ffmpeg.crop(video_clip, upper_half_x, upper_half_y, upper_half_w, upper_half_h)
# #             upper_half_clip = ffmpeg.output(upper_half_clip, os.path.join(upper_half_dir, '%05d.jpg'))
#             upper_half_clip = ffmpeg.output(upper_half_clip, audio, os.path.join(upper_half_dir, 'vid.mp4'))
#             ffmpeg.run(upper_half_clip, overwrite_output=True, quiet=True)
        
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_hp', type=str, default='../DATA/UpperBody/output_hp/bbox_hp.json')
    parser.add_argument('--video_dir', type=str, default='../DATA/UpperBody/video')
#     parser.add_argument('--output_dir', type=str, default='./Pilot/crop_frames')
    parser.add_argument('--output_dir', type=str, default='../DATA/UpperBody/crop_whole_video')
    args = parser.parse_args()

    crop_clip(args.bbox_hp, args.video_dir, args.output_dir)