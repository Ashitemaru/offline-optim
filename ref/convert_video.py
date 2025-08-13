import os, random


def convert_all(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, os.path.basename(video)[:-4] + ".mp4")

        # bitrate = random.sample(range(35, 40), 1)[0]
        # if video_path.endswith('264'):
        #     cmd = "ffmpeg -y -i {} -c:v h264_nvenc -b:v {}M -vtag avc1 {}".format(video_path, bitrate, output_path)
        # else:
        #     cmd = "ffmpeg -y -i {} -c:v h265_nvenc -b:v {}M -vtag hvc1 {}".format(video_path, bitrate, output_path)
        if video_path.endswith("264"):
            cmd = "ffmpeg -y -i {} -c:v copy -vtag avc1 {}".format(
                video_path, output_path
            )
        else:
            cmd = "ffmpeg -y -i {} -c:v copy -vtag hvc1 {}".format(
                video_path, output_path
            )
        print(cmd)

        if os.path.exists(output_path):
            continue
        os.system(cmd)


if __name__ == "__main__":
    # convert_all(r"D:\e2e_videos\tearing_results", r"D:\e2e_videos\tearing_results_converted")
    convert_all(
        r"D:\e2e_videos\framerate_results", r"D:\e2e_videos\framerate_results_converted"
    )
