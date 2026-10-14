import subprocess, os, sys

input_video = "/data/jobs/15d72d1857fe4a0f99255fd19fb5afae/artifacts/WhatsApp Video 2026-10-05 at 11.57.23 PM-1.mp4"
output_video = "/data/jobs/15d72d1857fe4a0f99255fd19fb5afae/artifacts/masking_video.mp4"
out_dir = "/tmp/compare_frames"
os.makedirs(out_dir, exist_ok=True)

# Extract frames at subtitle timestamps from both videos
timestamps = [1.0, 2.5, 3.5, 5.8, 6.5, 7.0, 8.0, 16.0, 17.5, 18.5, 20.5, 29.5, 38.0]

for ts in timestamps:
    ts_str = f"{ts:.1f}"
    # Input frame
    cmd_in = ["ffmpeg", "-y", "-ss", ts_str, "-i", input_video, "-vframes", "1", "-q:v", "2", f"{out_dir}/in_{ts_str}.jpg"]
    subprocess.run(cmd_in, stderr=subprocess.DEVNULL)
    # Output frame
    cmd_out = ["ffmpeg", "-y", "-ss", ts_str, "-i", output_video, "-vframes", "1", "-q:v", "2", f"{out_dir}/out_{ts_str}.jpg"]
    subprocess.run(cmd_out, stderr=subprocess.DEVNULL)

print("Frame extraction done")
# Build side-by-side comparison images
import subprocess
for ts in timestamps:
    ts_str = f"{ts:.1f}"
    in_f = f"{out_dir}/in_{ts_str}.jpg"
    out_f = f"{out_dir}/out_{ts_str}.jpg"
    cmp_f = f"{out_dir}/cmp_{ts_str}.jpg"
    if os.path.exists(in_f) and os.path.exists(out_f):
        subprocess.run([
            "ffmpeg", "-y",
            "-i", in_f, "-i", out_f,
            "-filter_complex",
            "[0:v]scale=478:850,drawtext=text='INPUT @" + ts_str + "s':x=10:y=10:fontsize=24:fontcolor=yellow[a];"
            "[1:v]scale=478:850,drawtext=text='OUTPUT @" + ts_str + "s':x=10:y=10:fontsize=24:fontcolor=cyan[b];"
            "[a][b]hstack=inputs=2",
            cmp_f
        ], stderr=subprocess.DEVNULL)
        print(f"  Built comparison at {ts_str}s")

print("All comparisons done")
print("Files in:", out_dir)
print(os.listdir(out_dir))

