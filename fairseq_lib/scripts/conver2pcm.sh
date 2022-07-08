# !/bin/sh
input=$1;
output=$2;
for file in `ls $input` 
    do 
        #echo "ffmpeg -i $input/$file -acodec pcm_s16le -ac 1 -ar 16000 $output/$file.wav";
        ffmpeg -i "$input/$file" -acodec pcm_s16le -ac 1 -ar 16000 "$output/$file.wav";
    done