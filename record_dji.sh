#!/bin/bash

# Configuration
OUTPUT_DIR="Videos"
RESOLUTION="1920x1080"
FRAMERATE="30"

# Camera 1 (Left)
CAM1_VIDEO="/dev/video2"
CAM1_AUDIO="alsa_input.usb-DJI_OsmoAction5pro_SN_8B72E2ED_123456789ABCDEF-01.analog-stereo"

# Camera 2 (Right)
CAM2_VIDEO="/dev/video4"
CAM2_AUDIO="alsa_input.usb-DJI_OsmoAction5pro_SN_FCEE12D0_123456789ABCDEF-01.analog-stereo"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Common FFmpeg settings
# High thread_queue_size to prevent frames from dropping while the other process initializes
FFMPEG_OPTS="-hide_banner -loglevel error -stats -thread_queue_size 4096 -rtbufsize 250M"
ENCODE_OPTS="-c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p -vsync cfr -c:a aac -b:a 128k -avoid_negative_ts make_non_negative"

echo "Starting Stereo Recording..."
echo "---------------------------------------------------"

# Capture timestamp L and launch CAM 1
TS_L=$(date +"%Y-%m-%d_%H-%M-%S-%3N")
FILE_L="$OUTPUT_DIR/${TS_L}_LEFT.mp4"
ffmpeg $FFMPEG_OPTS -f pulse -i "$CAM1_AUDIO" \
    -f v4l2 -input_format mjpeg -video_size "$RESOLUTION" -framerate "$FRAMERATE" -i "$CAM1_VIDEO" \
    $ENCODE_OPTS "$FILE_L" &
PID_L=$!

# Capture timestamp R and launch CAM 2
TS_R=$(date +"%Y-%m-%d_%H-%M-%S-%3N")
FILE_R="$OUTPUT_DIR/${TS_R}_RIGHT.mp4"
ffmpeg $FFMPEG_OPTS -f pulse -i "$CAM2_AUDIO" \
    -f v4l2 -input_format mjpeg -video_size "$RESOLUTION" -framerate "$FRAMERATE" -i "$CAM2_VIDEO" \
    $ENCODE_OPTS "$FILE_R" &
PID_R=$!

echo "CAM1 (L): Started at $TS_L"
echo "CAM2 (R): Started at $TS_R"
echo "---------------------------------------------------"
echo "RECORDING... Press [ENTER] to stop both."
echo "---------------------------------------------------"

# Wait for user input
read -r

echo "Stopping stereo recording..."
# Kill both pids at once to finalize as close together as possible
kill -SIGINT "$PID_L" "$PID_R"
wait "$PID_L" "$PID_R"

echo "Stereo capture finished."
echo "Final Files:"
echo "  $FILE_L"
echo "  $FILE_R"
