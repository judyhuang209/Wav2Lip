DIR="/home/judy/avspeech_preprocessed/subtest/*"
for d in $DIR ; do
	cd "$d" && echo $PWD && ffmpeg -i audio.wav -af aresample=resampler=soxr -ar 16000 -ac 1 audio.wav -y
done
