# HQ-Face-Dataset

## Install

- Prepare the environment with requirement.yaml
- Download [shape predictor model](https://www.dropbox.com/s/w565vfgik27bq2i/shape_predictor_68_face_landmarks.dat?dl=0) and put it in script folder.
- Download [human parsing model](https://www.dropbox.com/s/uh5io2ayyuxqg37/hp_model.zip?dl=0) and put it in script folder.

## Usage
- Put videos in DATA/video and name each video with youtube unique id.
- For each video, get a image of the target person's face, put them in DATA/info/face and name each image with youtube unique id.
- For each video, get the subtitle file, put them in DATA/info/subtitle and name each image with youtube unique id.
- run run.sh
