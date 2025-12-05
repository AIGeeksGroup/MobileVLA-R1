import multiprocessing.dummy as mp
import re
import subprocess
from os import listdir, mkdir


def extract_frames(videopath, dest, fps=1):

    try:
        mkdir(dest)
        print("creating " + dest + " subdirectory")
    except:
        print(dest + " subdirectory already exists")

    output = subprocess.call(
        [
            "ffmpeg",
            "-i",
            videopath,
            "-vf",
            "fps=" + str(fps),
            dest + "/%04d.jpg",
        ]
    )
    if output:
        print("Failed to extract frames")


def extract_all_frames():
    try:
        mkdir("/PATH_TO_DATA/NaVILA-Dataset/Human/raw_frames", exist_ok=True)
        print("creating frames subdirectory")
    except:
        print("frames subdirectory already exists")
    videos = listdir("/PATH_TO_DATA/NaVILA-Dataset/Human/videos")

    def eaf(vid):
        vid_id = re.match("(.*).mp4", vid)[1]
        subdir = "/PATH_TO_DATA/NaVILA-Dataset/Human/raw_frames/" + vid_id
        try:
            mkdir(subdir)
            extract_frames("/PATH_TO_DATA/NaVILA-Dataset/Human/videos/" + vid, subdir, fps=1)
        except FileExistsError:
            print(f"skipping {vid}")

    vids = [vid for vid in videos]
    p = mp.Pool(processes=8)
    p.map(eaf, vids)
    p.close()
    p.join()


if __name__ == "__main__":
    extract_all_frames()