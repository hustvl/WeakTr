#!/usr/bin/env bash
global_path='.'
data_dir=$global_path

year='2014'

print_usage() {
  printf "Usage: ..."
}
# usage: -y 2014 or 2017
while getopts 'y' flag; do
  case "${flag}" in
    y) year="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

cd $data_dir

mkdir coco
cd coco

mkdir images
cd images

echo "Downloading train and validation images"

# Download Images and annotations
wget -c http://images.cocodataset.org/zips/train${year}.zip
wget -c http://images.cocodataset.org/zips/val${year}.zip

# Unzip
echo "Unziping train folder"
unzip -q train${year}.zip
echo "Unziping val folder"
unzip -q val${year}.zip

echo "Deleting zip files"
rm -rf train${year}.zip
rm -rf val${year}.zip

echo "COCO data downloading over!!"

cd ..
echo "Downloading annotations"
wget -c http://images.cocodataset.org/annotations/annotations_trainval${year}.zip
unzip -q annotations_trainval${year}.zip
rm -rf annotations_trainval${year}.zip
echo "Done"

cd ..

python parse_coco.py --split train --year ${year} --to-voc12 false # support also year 2017
python parse_coco.py --split val --year ${year} --to-voc12 false # support also year 2017
