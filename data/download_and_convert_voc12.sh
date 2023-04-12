#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_tfrecord.py
#     - download_and_convert_voc2012.sh
#     - parse_voc12.py
#     + voc12
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
# set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./voc12"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack VOC 2012 dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}
  local DESTINATION=${3}
  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  case "${FILENAME}" in
  *.tar ) 
          tar -xvf "${FILENAME}" -C "${DESTINATION}"
          ;;
  *.tgz )
          tar zxvf "${FILENAME}" -C "${DESTINATION}"
          ;;
  *.zip )
          unzip "${FILENAME}" -d "${DESTINATION}"
          ;;
  esac
}


# Download the images.
BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
FILENAME="VOCtrainval_11-May-2012.tar"

download_and_uncompress "${BASE_URL}" "${FILENAME}" "."

# Download the test images.
BASE_URL="http://pjreddie.com/media/files/" # mirror, need to be logged in to download from official server
FILENAME_TEST="VOC2012test.tar"

download_and_uncompress "${BASE_URL}" "${FILENAME_TEST}" "."

wget -P "./VOCdevkit/VOC2012/ImageSets/Segmentation" "https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt"

rm ${FILENAME} # remove tar file
rm ${FILENAME_TEST} # remove tar file

