#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

DATA_DIR="data/PuzzleIOI"

mkdir -p $DATA_DIR

# username and password input
echo -e "\nYou need to register at https://puzzleavatar.is.tue.mpg.de/, and input the username and password below."
read -p "Username (PuzzleAvatar):" username
read -p "Password (PuzzleAvatar):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading PuzzleIOI 3D scans and SMPL-X fits used for training and evaluation..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=puzzleavatar&sfile=fitting.zip&resume=1' -O $DATA_DIR/fitting.zip --no-check-certificate --continue
unzip $DATA_DIR/fitting.zip -d $DATA_DIR

echo -e "\nDownloading PuzzleIOI multi-view captures used for training..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=puzzleavatar&sfile=puzzle_capture.zip&resume=1' -O $DATA_DIR/puzzle_capture.zip --no-check-certificate --continue
unzip $DATA_DIR/puzzle_capture.zip -d $DATA_DIR
mv $DATA_DIR/puzzle_cam $DATA_DIR/puzzle_capture

echo -e "\nDownloading PuzzleIOI ground-truth renderings used for evaluation..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=puzzleavatar&sfile=gt_4views.zip&resume=1' -O $DATA_DIR/gt_4views.zip --no-check-certificate --continue
unzip $DATA_DIR/gt_4views.zip -d $DATA_DIR
mv $DATA_DIR/rendering_4views $DATA_DIR/gt_4views

echo -e "\nDownloading PuzzleIOI subject lists (full, ablation-test)..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=puzzleavatar&sfile=subjects_all.txt&resume=1' -O $DATA_DIR/subjects_all.txt --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=puzzleavatar&sfile=subjects_test.txt&resume=1' -O $DATA_DIR/subjects_test.txt --no-check-certificate --continue

echo "Great jobs! Now you can waste your time on training and evaluation!"
