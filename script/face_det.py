import argparse
import math
import cv2
import dlib
import numpy as np
import os
import csv

# frame_30fps_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/tmp/frame_30fps'
# frame_2fps_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/tmp/frame_2fps'
# scene_detect_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/tmp/scene'
# face_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/info/face'

# frame_30fps_path = './frame_30fps'
# frame_2fps_path = './frame_2fps'
# face_path = './face'
# scene_detect_path = './fuwuqi'

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def check_data(data_dir):
    video = [i.split(".")[0] for i in os.listdir(os.path.join(data_dir, "video"))]
    frame = [i for i in os.listdir(os.path.join(data_dir, "tmp/frame_2fps"))]
    scene = [i.split("-Scenes")[0] for i in os.listdir(os.path.join(data_dir, "tmp/scene"))]
    face = [i.split(".")[0] for i in os.listdir(os.path.join(data_dir, "info/face"))]

    for v in video:
        assert v in frame, "ERROR: Frame extraction failed for "+ v
        assert v in scene, "ERROR: Scene detect failed for "+ v
        assert v in face, "ERROR: No face template for "+ v


def get_quality(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


def distance(a, b):
    a, b = np.array(a), np.array(b)
    sub = np.sum((a - b) ** 2)
    add = (np.sum(a ** 2) + np.sum(b ** 2)) / 2.
    return sub / add


def get_feature(p):
    img = cv2.imread(p)
    scale_percent = img.shape[0] / 512  # percent of original size
    width = int(img.shape[1] / scale_percent)
    height = int(img.shape[0] / scale_percent)
    dim = (width, height)
    img_small = cv2.resize(img, dim)
    dets = detector(img_small)
    face_vector_list = []
    shape_list = []
    if len(dets) == 1:
        for d in dets:
            shape = predictor(img_small, d)
            face_vector = facerec.compute_face_descriptor(img_small, shape)
            face_vector_list.append(face_vector)
            shape_list.append(shape)
    return face_vector_list, shape_list, scale_percent, dets


def classifier(a, b, t=0.15):
    if distance(a, b) <= t:
        ret = True
    else:
        ret = False
    return ret


def mainpart(face_path, result_path, frame_2fps_path, landmark_path, data_dir):
    print("start checking")
    check_data(data_dir)
    print("end checking")
    period = {}
    g = os.walk(frame_2fps_path)
    with open(landmark_path, 'w+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['uid']
        header += ['frame_2fps']
        header += ['x1', 'y1']
        header += ['x2', 'y2']
        for i in range(68):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]
        csv_writer.writerow(header)

        for path, dir_list, file_list in g:
            for video in dir_list:
                print(video)
                fpath = os.path.join(frame_2fps_path, video)
                face_name = video + '.png'
                target_face = os.path.join(face_path, face_name)
                face_gt_feature, _, _, _ = get_feature(target_face)
                if len(face_gt_feature) != 1:
                    print(video + ': cannot recognize the target face')
                    continue
                face_gt_feature = face_gt_feature[0]
                gg = os.walk(fpath)
                video_period = []
                for p, dl, fl in gg:
                    fl.sort()
                    for frame in fl:
                        row = [video]
                        frame_path = os.path.join(fpath, frame)
                        feature, shapes, scale, faces = get_feature(frame_path)
                        if len(feature) == 1:
                            f = feature[0]
                            if classifier(f, face_gt_feature) is True:
                                t = int(frame.split('.')[0].split('_')[1]) * 15
                                row += [int(frame.split('.')[0].split('_')[1])]
                                video_period.append(t)
                                row += [int(faces[0].left() * scale), int(faces[0].top() * scale),
                                        int(faces[0].right() * scale), int(faces[0].bottom() * scale)]
                                for i in range(68):
                                    part_i_x = int(shapes[0].part(i).x * scale)
                                    part_i_y = int(shapes[0].part(i).y * scale)
                                    row += [part_i_x, part_i_y]
                                csv_writer.writerow(row)

                    period[video] = video_period
                    print(len(video_period))

    # txt_path = './time_stamp.txt'
    with open(result_path, 'w+') as txt:
        for k in period.keys():
            p = period[k]
            t0 = p[0]
            for i in range(1, len(p)):
                if p[i] != p[i - 1] + 15:
                    start_frame = t0
                    end_frame = p[i - 1]
                    temp = k + ' ' + str(start_frame) + ' ' + str(end_frame)
                    txt.write(temp)
                    txt.write('\r\n')
                    t0 = p[i]
            start_frame = t0
            end_frame = p[-1]
            temp = k + ' ' + str(start_frame) + ' ' + str(end_frame)
            txt.write(temp)
            txt.write('\r\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--face_path', type=str)
    parser.add_argument('--frame_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--landmark_dir', type=str)
    args = parser.parse_args()

    frame_2fps_path = args.frame_path
    face_path = args.face_path
    result_path = args.output_dir
    landmark_path = args.landmark_dir
    data_dir = args.data_dir

    mainpart(face_path, result_path, frame_2fps_path, landmark_path, data_dir)
