#!/bin/bash
for file in `ls $1/*`
do
    var=$(basename $file)
    name=${var%.*}
	echo "$name"
	PTHH=$2/$name
	echo "$PTHH"
	if [ ! -d "$PTHH"  ];then
		echo "---------------------------------"
		mkdir "$PTHH"
		ffmpeg -i "$file" -f image2 -vf fps=2 -qscale:v 2 "$PTHH/img_%05d.jpg"
	fi
done