import ffmpeg

input_file = '6-6-24 john blackhawk real time.mov'
output_file = '6-6-24 john blackhawk real time.mp4'

ffmpeg.input(input_file).output(output_file, vcodec='libx264', acodec='aac').run()
