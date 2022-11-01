import os
import argparse
import json
import logging
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import matplotlib.patches as patches
from tqdm import tqdm
import time

from hp_model.net.pspnet import PSPNet


models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

# parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
# parser.add_argument('image_path', type=str, help='Path to image')
# parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
# parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
# parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")
# args = parser.parse_args()



def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)


def show_image(img, pred, uid, ori_w, ori_h):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    
    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))

    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=0.7, )
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(2.3, (j + 0.45), lab, ha='left', va='center', )

    plt.savefig(fname="./result/"+uid+".jpg")
    plt.show()


def draw_bbox(img, boxes, uid, t, label=None, classes=None):
    '''
    Args:
      img: narrary image
      boxes: Nx4 [[xmin, ymin, xmax, ymax],[.. ..]]
      label: [c1, c2 ,c3,....] 1xN
      classes: ['bird', 'cat', 'dog', ....]
    '''
    plt.imshow(img)
    color = ['r', 'g', 'b', 'y']
    wh = boxes[:,2:4] - boxes[:,0:2]
    for i in range(len(boxes)):
        currentAxis=plt.gca()
        rect=patches.Rectangle((boxes[i,0], boxes[i,1]),wh[i, 0],wh[i, 1],linewidth=1.5,edgecolor=color[i],facecolor='none')
        currentAxis.add_patch(rect)
        plt.text(boxes[i,0], boxes[i,1], classes[label[i]-1], color='white',fontsize=16)
    if not os.path.exists("./Pilot/tmp/bbox_visual/"+uid):
        os.mkdir("./Pilot/tmp/bbox_visual/"+uid)
    plt.savefig(fname="./Pilot/tmp/bbox_visual/"+uid+"/"+str(t)+".jpg")
    plt.cla()
#     plt.show()

    
def denormalize(img, mean, std):
    c, _, _ = img.shape
    for idx in range(c):
        img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
    return img
    
    
def get_bbox(net, img_path, uid):
    
    # ------------ load image ------------ #
    data_transform = get_transform()
    img = Image.open(img_path)
    w, h = img.width, img.height
    img = data_transform(img)
    img = img.cuda()

    # --------------- inference --------------- #
#     t1_s = time.time()
    with torch.no_grad():
        pred, _ = net(img.unsqueeze(dim=0))
        pred = pred.squeeze(dim=0)
        pred = pred.cpu().numpy().transpose(1, 2, 0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
#         show_image(img, pred, vid, w, h)
#     t1_e = time.time()
#     t2_s = time.time()
    face_bbox, upper_half_bbox, upper_body_bbox, upper_full_bbox, arms_label = get_region_bbox2(pred)
#     t2_e = time.time()
    boxes = np.array([face_bbox, upper_half_bbox, upper_body_bbox, upper_full_bbox], dtype=float)
#     label = [1, 2, 3]
#     classes = ['face', 'upper half', 'upper body']
#     label = [1]
#     classes = ['upper half']
#     boxes = np.array([upper_half_bbox], dtype=float)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     h, w, _ = pred.shape
#     img = img.transpose(1, 2, 0).reshape((h, w, 3))
    boxes = enlarge_bbox(boxes)
#     draw_bbox(img, boxes, uid, label, classes)
#     print((t1_e-t1_s), (t2_e-t2_s))
    return boxes, pred, arms_label

def show_bboxes(img_path, boxes, uid, t):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = [1, 2, 3, 4]
    classes = ['face', 'upper half', 'upper body', 'upper_full_bbox']
    draw_bbox(img, boxes, uid, t, label, classes)
    
    
def get_region_bbox(img):
    # get bbox
# classes = np.array(('Background',  # always index 0
#                         'Hat'1, 'Hair'2, 'Glove'3, 'Sunglasses'4,
#                         'UpperClothes'5, 'Dress'6, 'Coat'7, 'Socks'8,
#                         'Pants'9, 'Jumpsuits'10, 'Scarf'11, 'Skirt'12,
#                         'Face'13, 'Left-arm'14, 'Right-arm'15, 'Left-leg'16,
#                         'Right-leg'17, 'Left-shoe'18, 'Right-shoe'19,))
    upper_body = [5, 6, 7, 10, 11]
    arms = [3, 14, 15]
    head = [1, 4, 13]
    hair = [2]
    
    # [x1, y1, x2, y2]
    face_bbox = [255, 255, 0, 0]
    upper_half_bbox = [255, 255, 0, 0]
    upper_body_bbox = [255, 255, 0, 0]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # face
            if img[y, x] in head:
                face_bbox[0] = min(x, face_bbox[0])
                face_bbox[1] = min(y, face_bbox[1])
                face_bbox[2] = max(x, face_bbox[2])
                face_bbox[3] = max(y, face_bbox[3])
            elif img[y, x] in hair:
                face_bbox[1] = min(y, face_bbox[1])
                face_bbox[0] = min(x, face_bbox[0])
                face_bbox[2] = max(x, face_bbox[2])
                
            # upper half
            if img[y, x] in upper_body:
                upper_half_bbox[0] = min(x, upper_half_bbox[0])
                upper_half_bbox[1] = min(y, upper_half_bbox[1])
                upper_half_bbox[2] = max(x, upper_half_bbox[2])
                upper_half_bbox[3] = max(y, upper_half_bbox[3])
                
            # upper body
            if img[y, x] in head+arms+upper_body+hair:
                upper_body_bbox[0] = min(x, upper_body_bbox[0])
                upper_body_bbox[1] = min(y, upper_body_bbox[1])
                upper_body_bbox[2] = max(x, upper_body_bbox[2])
                upper_body_bbox[3] = max(y, upper_body_bbox[3])
#             elif img[y, x] in hair:
#                 upper_body_bbox[1] = min(y, upper_body_bbox[1])
                
    upper_half_bbox[3] = int(upper_half_bbox[1] + (upper_half_bbox[3]-upper_half_bbox[1])/3)
    upper_half_bbox[1] = face_bbox[1]
    upper_half_bbox[0] = min(upper_half_bbox[0], face_bbox[0])
    upper_half_bbox[2] = max(upper_half_bbox[2], face_bbox[2])
                
    return face_bbox, upper_half_bbox, upper_body_bbox

def get_region_bbox2(img):
    hash_map = dict()
    upper_body = [5, 6, 7, 10, 11]
    arms = [3, 14, 15]
    head = [1, 4, 13]
    hair = [2]

    hash_map["upper_body"] = []
    hash_map["head"] = []
    hash_map["arms"] = []
    hash_map["hair"] = []
    for x in range(0, img.shape[0], 3):
        for y in range(0, img.shape[1], 3):
            if img[y, x] in upper_body:
                hash_map["upper_body"].append([y, x])
            if img[y, x] in arms:
                hash_map["arms"].append([y, x])    
            if img[y, x] in head:
                hash_map["head"].append([y, x])
            if img[y, x] in hair:
                hash_map["hair"].append([y, x])
             
    
    def pos2box(pos):
        pmin = np.min(pos, axis=0)
        xmin, ymin = pmin[1], pmin[0]
        pmax = np.max(pos, axis=0)
        xmax, ymax = pmax[1], pmax[0]
        return [xmin, ymin, xmax, ymax]
    
    
    head_pos = np.array(hash_map["head"])
    arms_pos = np.array(hash_map["arms"])
    hair_pos = np.array(hash_map["hair"])
    upper_body_pos = np.array(hash_map["upper_body"])
    
    if head_pos.size == 0:
        head_box = [255, 255, 0, 0]
    else:
        head_box = pos2box(head_pos)
        
    if arms_pos.size == 0:
        arms_box = [255, 255, 0, 0]
    else:
        arms_box = pos2box(arms_pos)
        
    if hair_pos.size == 0:
        hair_box = [255, 255, 0, 0]
    else:
        hair_box = pos2box(hair_pos)
        
    if upper_body_pos.size == 0:
        upper_body_box = [255, 255, 0, 0]
    else:
        upper_body_box = pos2box(upper_body_pos)
    
    face_bbox = [255, 255, 0, 0]
    upper_half_bbox = [255, 255, 0, 0]
    upper_body_bbox = [255, 255, 0, 0]
    
    
#     face_bbox = [min(hair_box[0], head_box[0]), 
#                  min(hair_box[1], head_box[1]), 
#                  max(hair_box[2], head_box[2]), 
#                  head_box[3]]
#     upper_half_bbox[3] = int(upper_body_box[1] + (upper_body_box[3]-upper_body_box[1])/3)
#     upper_half_bbox[1] = face_bbox[1]
#     upper_half_bbox[0] = min(upper_body_box[0], face_bbox[0])
#     upper_half_bbox[2] = max(upper_body_box[2], face_bbox[2])
#     upper_body_bbox = [min(head_box[0], arms_box[0], hair_box[0], upper_body_box[0]),
#                        min((head_box[1], arms_box[1], hair_box[1], upper_body_box[1])),
#                        max(head_box[2], arms_box[2], hair_box[2], upper_body_box[2]),
#                        max(head_box[3], arms_box[3], hair_box[3], upper_body_box[3])]
    face_bbox = [min(hair_box[0], head_box[0]), 
                 min(hair_box[1], head_box[1]), 
                 max(hair_box[2], head_box[2]), 
                 head_box[3]]
    upper_half_bbox[3] = int(max(upper_body_box[1], face_bbox[3]) + (upper_body_box[3]-max(upper_body_box[1], face_bbox[3]))/3)
    upper_half_bbox[1] = face_bbox[1]
#     print(img[upper_half_bbox[3], :, 0].shape)
    line = img[upper_half_bbox[3], :, 0].tolist()
    for i, v in enumerate(line):
        if v in upper_body+head+hair:
            upper_half_bbox[0] = i
            break
    for i, v in enumerate(line[::-1]):
        if v in upper_body+head+hair:
            upper_half_bbox[2] = 255-i
            break
    upper_half_bbox[0] = min(upper_half_bbox[0], face_bbox[0])
    upper_half_bbox[2] = max(upper_half_bbox[2], face_bbox[2])
#     upper_half_bbox[0] = min(upper_body_box[0], face_bbox[0])
#     upper_half_bbox[2] = max(upper_body_box[2], face_bbox[2])
    upper_body_bbox = [min(head_box[0], hair_box[0], upper_body_box[0]),
                       min((head_box[1], hair_box[1], upper_body_box[1])),
                       max(head_box[2], hair_box[2], upper_body_box[2]),
                       max(head_box[3], hair_box[3], upper_body_box[3])]

    upper_full_bbox = [min(head_box[0], arms_box[0], hair_box[0], upper_body_box[0]),
                       min((head_box[1], arms_box[1], hair_box[1], upper_body_box[1])),
                       max(head_box[2], arms_box[2], hair_box[2], upper_body_box[2]),
                       max(head_box[3], arms_box[3], hair_box[3], upper_body_box[3])]
    
    arms_label = {"face_bbox":0, "upper_half_bbox":0, "upper_body_bbox":0, "upper_full_bbox":0}
    
    def is_inbox(point, box):
        if point[1] > box[0] and point[1] < box[2] and point[0] > box[1] and point[0] < box[3]:
            return True
        else:
            return False
    
    for point in hash_map["arms"]:
        if arms_label["face_bbox"] == 0:
            if is_inbox(point, face_bbox):
                arms_label["face_bbox"] = 1
        if arms_label["upper_half_bbox"] == 0:
            if is_inbox(point, upper_half_bbox):
                arms_label["upper_half_bbox"] = 1
        if arms_label["upper_body_bbox"] == 0:
            if is_inbox(point, upper_body_bbox):
                arms_label["upper_body_bbox"] = 1
        if arms_label["upper_full_bbox"] == 0:
            if is_inbox(point, upper_full_bbox):
                arms_label["upper_full_bbox"] = 1
        
    
#     return head_box, upper_body_box, arms_box
    return face_bbox, upper_half_bbox, upper_body_bbox, upper_full_bbox, arms_label
    
def enlarge_bbox(bboxes, ratio=1.1):
    for bbox in bboxes:
        # bbox = [xmin, ymin, xmax, ymax]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        delta_w = (width * (ratio-1))/2
        delta_h = (height * (ratio-1))/2
        
        bbox[0] = max(0, bbox[0]-delta_w)
        bbox[1] = max(0, bbox[1]-delta_h)
        bbox[2] = min(255, bbox[2]+delta_w)
        bbox[3] = min(255, bbox[3]+delta_w)
#         break
    return bboxes

def recover_size(bboxes, w, h):
    for box in bboxes:
        box[0] = int(box[0]/256 * w)
        box[1] = int(box[1]/256 * h)
        box[2] = int(box[2]/256 * w)
        box[3] = int(box[3]/256 * h)
    return bboxes

# frame_2fps_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/tmp/frame_2fps'
# frame_2fps_path = './frame_2fps'
# json_path = '/home/hqface/HQ-Face-Video-Dataset/DATA/tmp/clip_info.json'
# json_path = './fuwuqi/clip_info.json'

# detector = dlib.get_frontal_face_detector()
# predictor_path = 'shape_predictor_68_face_landmarks.dat'
# predictor = dlib.shape_predictor(predictor_path)
# face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)


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
    img_scale = cv2.resize(img, dim)
    dets = detector(img_scale)
    # face_vector_list = []
    face_q_list = []
    bbox_list = []
    for i, d in enumerate(dets):
        # shape = predictor(img_scale, d)
        y1 = max(d.top(), 0) * scale_percent
        y2 = max(d.bottom(), 0) * scale_percent
        x1 = max(d.left(), 0) * scale_percent
        x2 = max(d.right(), 0) * scale_percent
        # face_vector = facerec.compute_face_descriptor(img_scale, shape)
        # face_vector_list.append(face_vector)
        face = img[int(y1):int(y2), int(x1):int(x2), :]
        face_q = get_quality(face)
        face_q_list.append(face_q)
        bbox = [x1, y1, x2, y2]
        bbox_list.append(bbox)
    return face_q_list, bbox_list


def classifier(a, b, t=0.15):
    if distance(a, b) <= t:
        ret = True
    else:
        ret = False
    return ret


def mainpart(json_path, result_path, frame_2fps_path):
    models_path = "./hp_model/checkpoints"
    backend = "densenet"
    num_classes = 20

    # --------------- model --------------- #
    snapshot = os.path.join(models_path, backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, backend)
    net.eval()
    
    bbox_list = []
    with open(json_path) as f:
        reader = json.load(f)
        for uid in tqdm(reader.keys(), total=len(reader)):
            tqdm.write(uid)
            temp_dict = {'uid': uid}
            clips = []
            clip_info = reader[uid]
            v_path = os.path.join(frame_2fps_path, uid)

            # face_name = uid + '.png'
            # target_face = os.path.join(face_path, face_name)
            # face_gt_feature, _, _ = get_feature(target_face)
            # face_gt_feature = face_gt_feature[0]
            for n_clip, c in enumerate(clip_info):
                clip_result = {}
                bbox_l = []
                clip_pred = []
                start_time = float(c['start'])
                end_time = float(c['end'])
                start_frame = int(start_time * 2)
                end_frame = int(end_time * 2)
                clip_result['start_time'] = start_time
                clip_result['end_time'] = end_time
                clip_result['text'] = c['text']
                for t in range(start_frame, end_frame):
                    # print(t)
                    bbox_dict = {}
                    t_name = 'img_' + str(t).zfill(5) + '.jpg'
                    t_path = os.path.join(v_path, t_name)
                    if 'video_height' not in temp_dict.keys():
                        img = cv2.imread(t_path)
                        temp_dict['video_width'] = img.shape[1]
                        temp_dict['video_height'] = img.shape[0]
                    
                    # boxes = [face_bbox, upper_half_bbox, upper_body_bbox]
                    trans_bboxes, pred, arms_label = get_bbox(net, t_path, uid)
                    clip_pred.append(pred)
                    bboxes = recover_size(trans_bboxes, img.shape[1], img.shape[0])
#                     show_bboxes(t_path, bboxes, uid, t)
                    bboxes = bboxes.tolist()
                    bbox_dict['frame'] = t
                    bbox_dict['face_bbox'] = bboxes[0]
                    bbox_dict['upper_half_bbox'] = bboxes[1]
                    bbox_dict['upper_body_bbox'] = bboxes[2]
                    bbox_dict['upper_full_bbox'] = bboxes[3]
                    bbox_dict['arms_label'] = arms_label
                    
#                     quality, bbox = get_feature(t_path)
#                     b = bbox[0]
#                     q = quality[0]
#                     x1 = b[0]
#                     y1 = b[1]
#                     x2 = b[2]
#                     y2 = b[3]
#                     bbox_dict['frame'] = t
#                     bbox_dict['x1'] = x1
#                     bbox_dict['y1'] = y1
#                     bbox_dict['x2'] = x2
#                     bbox_dict['y2'] = y2
                    # bbox_dict['quality'] = q
                    # print(x1)
                    # print(x2)
                    # print(y1)
                    # print(y2)
                    # print(bbox_dict)
                    bbox_l.append(bbox_dict)
                clip_result['bbox'] = bbox_l
                clips.append(clip_result)
                hp_output_path = os.path.join(result_path, "human_parsing_pred", uid)
                if not os.path.exists(os.path.join(result_path, "human_parsing_pred")):
                    os.mkdir(os.path.join(result_path, "human_parsing_pred"))
                if not os.path.exists(hp_output_path):
                    os.mkdir(hp_output_path)
                np.save(os.path.join(hp_output_path, str(n_clip)+".npy"), np.array(clip_pred))

            temp_dict['clips_info'] = clips
            bbox_list.append(temp_dict)
    
    with open(os.path.join(result_path, "human_parsing_bbox.json"), 'w') as r:
        json.dump(bbox_list, r)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_info', type=str, default='./DATA/tmp/clip_info.json')
    parser.add_argument('--frame_path', type=str, default='./DATA/tmp/frame_2fps')
    parser.add_argument('--output_dir', type=str, default='./DATA/tmp')
    args = parser.parse_args()

    frame_2fps_path = args.frame_path
    json_path = args.clip_info
    result_path = args.output_dir

    mainpart(json_path, result_path, frame_2fps_path)
