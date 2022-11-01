#!/bin/bash
for file in `ls $1/*`
do
    var=$(basename $file)
    name=${var%.*}
	echo "$name"
	PTHH=$2/scene
	FIND=$2/scene/${name}-Scenes.csv
	echo "$FIND"
	if [ ! -f "$FIND"  ];then
		scenedetect --input $file --output $PTHH detect-content --threshold 12 list-scenes
	fi
done