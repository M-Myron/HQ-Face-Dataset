import argparse
import math
import os
import csv


def mainpart(face_time_path, result_path, scene_detect_path):
    period = {}
    while True:
        if os.path.exists(face_time_path):
            break
    with open(face_time_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            uid = line.split(' ')[0]
            start = line.split(' ')[1]
            end = line.split(' ')[2]
            if uid not in period.keys():
                period[uid] = []
            tmp = [start, end]
            period[uid].append(tmp)

    with open(result_path, 'w+') as txt:
        for k in period.keys():
            p = period[k]
            sd_name = k + '-Scenes.csv'
            sd_path = os.path.join(scene_detect_path, sd_name)
            read = csv.reader(open(sd_path))
            scene_time = [0]
            l = 0
            for line in read:
                if l == 0 or l == 1:
                    l += 1
                    continue
                else:
                    scene_end = math.floor(float(line[6]) * 30)
                    scene_time.append(scene_end)
                l += 1
            scene_time.append(1e10)

            for se in p:
                start_frame = int(se[0])
                end_frame = int(se[1])
                for idx in range(len(scene_time)-1):
                    if scene_time[idx] <= start_frame <= scene_time[idx + 1]:
                        if scene_time[idx + 1] >= end_frame:
                            temp = k + ' ' + str(start_frame) + ' ' + str(end_frame)
                            txt.write(temp)
                            txt.write('\r\n')
                            break
                        else:
                            for iidx in range(len(scene_time)-1):
                                if scene_time[iidx] <= end_frame <= scene_time[iidx + 1]:
                                    for iii in range(idx + 1, iidx + 1):
                                        if iii == idx + 1:
                                            temp = k + ' ' + str(start_frame) + ' ' + str(scene_time[iii])
                                            txt.write(temp)
                                            txt.write('\r\n')
                                        else:
                                            temp = k + ' ' + str(scene_time[iii-1]) + ' ' + str(scene_time[iii])
                                            txt.write(temp)
                                            txt.write('\r\n')
                                    temp = k + ' ' + str(scene_time[iidx]) + ' ' + str(end_frame)
                                    txt.write(temp)
                                    txt.write('\r\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_time_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--scene_detect_dir', type=str)
    args = parser.parse_args()

    face_time_path = args.face_time_path
    result_path = args.output_dir
    scene_detect_path = args.scene_detect_dir

    mainpart(face_time_path, result_path, scene_detect_path)
