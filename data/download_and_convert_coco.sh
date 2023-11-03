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
