#!/bin/bash
mkdir -p index

ls video | awk -F. '{print $1" "i"/video/"$0}' i=`pwd` > index/uid2path
ls info/face | awk -F. '{print $1" "i"/info/face/"$0}' i=`pwd` > index/uid2face
ls info/subtitle | awk -F. '{print $1" "i"/info/subtitle/"$0}' i=`pwd` > index/uid2sub
