import os
import random
import numpy as np
import csv

root_path = "D:\e2e_videos"


def frame_rate_simulate(
    start_no,
    frame_no,
    drop_windows_size,
    drop_windows_interval,
    drop_frame_num,
    drop_frame_mode,
    input_video,
):
    if drop_frame_num > drop_windows_size:
        print("drop num is larger than drop windows size!")
        return

    print("drop_windows_size:", drop_windows_size)
    print("drop_windows_interval:", drop_windows_interval)
    print("drop_frame_num:", drop_frame_num)
    print("drop_frame_mode:", drop_frame_mode)

    output_path = os.path.join(
        root_path,
        "framerate_results",
        "%s_%d_%d_%d_%d_%d.264"
        % (
            os.path.basename(input_video)[:-4],
            start_no + frame_no,
            drop_windows_size,
            drop_windows_interval,
            drop_frame_num,
            drop_frame_mode,
        ),
    )

    if os.path.exists(output_path):
        return

    # Open the CSV file in write mode
    file_path = os.path.join(root_path, "framerate_simulator", "frame_flag.csv")
    with open(file_path, "w", newline="") as f:
        # Create a CSV writer object
        writer = csv.writer(f)
        for i in range(start_no):
            writer.writerow([i, 1])

        writer.writerow([start_no, 1])
        windows_num = np.ceil(frame_no / drop_windows_size).astype(int)
        for i in range(windows_num):
            if i % drop_windows_interval == 0:  ## windows_interval
                n = drop_frame_num
                L1 = random.sample(range(0, drop_windows_size - 1), drop_frame_num)
                random_array = np.array(L1)
                for j in range(i * drop_windows_size, (i + 1) * drop_windows_size):
                    if j == frame_no - 1:
                        continue
                    if drop_frame_mode == 0:  ## continous mode
                        if n > 0:
                            writer.writerow([j + 1 + start_no, 0])
                            n -= 1
                        else:
                            writer.writerow([j + 1 + start_no, 1])
                    if drop_frame_mode == 1:  ## homogeneous mode
                        inter_interval = drop_windows_size // drop_frame_num
                        if (j - i * drop_windows_size) % inter_interval == 0 and n > 0:
                            n -= 1
                            writer.writerow([j + 1 + start_no, 0])
                        else:
                            writer.writerow([j + 1 + start_no, 1])
                    if drop_frame_mode == 2:  ## random mode
                        if (j - i * drop_windows_size) in random_array:
                            writer.writerow([j + 1 + start_no, 0])
                        else:
                            writer.writerow([j + 1 + start_no, 1])
            else:
                for j in range(i * drop_windows_size, (i + 1) * drop_windows_size):
                    if j == frame_no - 1:
                        continue
                    writer.writerow([j + 1 + start_no, 1])

    bitrate = random.sample(range(35, 40), 1)[0]
    cmd = "D:\\e2e_videos\\framerate_simulator\\demo.exe -input_seq {} -output_bin {} -color_space 3 -width 1920 -height 1080 -framerate 60 -bitrate {} -max_bitrate {} -encoder_type 0 -codec_type 16 -gpu_index 0 -gop_size 60 -ref_frames 3 -black_frame_detect 0 -business_id 1 -data_analysis 1 -vmaf 0 -vmaf_thread_num 0 > D:\\e2e_videos\\framerate_simulator\\anchor_264.log".format(
        input_video, output_path, bitrate * 1000, bitrate * 1000
    )
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    # 4 dimension
    start_no = 60
    frame_no = 900 - start_no

    input_videos = [
        # r"E:\e2e_videos\Others\yuv\LostArk_1920_1080_chunk_4.yuv",
        # r"E:\e2e_videos\Others\yuv\LostArk_1920_1080_chunk_5.yuv",
        # r"D:\e2e_videos\Valorant\yuv\Valorant_1920_1080_chunk_11.yuv",
        # r"D:\e2e_videos\Valorant\yuv\Valorant_1920_1080_chunk_24.yuv",
        # r"D:\e2e_videos\Yuanshen\yuv\Yuanshen_1920_1080_chunk_4.yuv",
        # r"D:\e2e_videos\Yuanshen\yuv\Yuanshen_1920_1080_chunk_10.yuv",
        # r"E:\e2e_videos\Others\yuv\LoL_1920_1080_chunk_1.yuv",
        # r"E:\e2e_videos\Others\yuv\LoL_1920_1080_chunk_2.yuv",
        # r"E:\e2e_videos\Others\yuv\DNF_1920_1080_chunk_1.yuv",
        # r"E:\e2e_videos\Others\yuv\DNF_1920_1080_chunk_2.yuv",
        # r"E:\e2e_videos\Others\yuv\NBA2K_1920_1080_chunk_1.yuv",
        # r"E:\e2e_videos\Others\yuv\NBA2K_1920_1080_chunk_3.yuv"
        # r"D:\e2e_videos\Nizhan\yuv\Nizhan_1920_1080_chunk_13.yuv",
        # r"D:\e2e_videos\Nizhan\yuv\Nizhan_1920_1080_chunk_15.yuv",
        # r"D:\e2e_videos\StartRail\yuv\StarRail_1920_1080_chunk_8.yuv",
        # r"D:\e2e_videos\StartRail\yuv\StarRail_1920_1080_chunk_10.yuv",
        # r"D:\e2e_videos\Others\yuv\lol_1920_1080_chunk_4.yuv",
        # r"D:\e2e_videos\Others\yuv\nba2k_1920_1080_chunk_3.yuv"
        # r"D:\e2e_videos\Others\yuv\DevilMayCry5_1920x1080_60fps.yuv",
        # r"D:\e2e_videos\Others\yuv\KOFXIV_1920x1080_60fps.yuv"
    ]

    # drop_windows_sizes = [18]     # 18 frames = 300ms
    # drop_windows_intervals = [1, 2, 3]  # windows interval
    # drop_frame_nums = [1, 2, 3, 4, 5, 6]       # drop frame num in one windows
    # drop_frame_modes = [0, 1, 2]         # drop mode: 0~continous  1~homogeneous  2~random

    # drop_windows_sizes = [60]     # 18 frames = 300ms
    # drop_windows_intervals = [1]  # windows interval
    # drop_frame_nums = [5, 10, 15]       # drop frame num in one windows
    # drop_frame_modes = [1]         # drop mode: 0~continous  1~homogeneous  2~random

    # drop_windows_sizes = [15, 30, 60]     # 18 frames = 300ms
    # drop_windows_intervals = [1]  # windows interval
    # drop_frame_nums = [1, 2, 3]       # drop frame num in one windows
    # drop_frame_modes = [0]         # drop mode: 0~continous  1~homogeneous  2~random

    # drop_windows_sizes = [12]     # 18 frames = 300ms
    # drop_windows_intervals = [1]  # windows interval
    # drop_frame_nums = [2]       # drop frame num in one windows
    # drop_frame_modes = [0]         # drop mode: 0~continous  1~homogeneous  2~random

    # drop_windows_sizes = [60]     # 18 frames = 300ms
    # drop_windows_intervals = [1]  # windows interval
    # drop_frame_nums = [10]       # drop frame num in one windows
    # drop_frame_modes = [2]         # drop mode: 0~continous  1~homogeneous  2~random

    params = [
        [[60], [1], [0], [0]],
        [[60], [1], [1, 2, 3], [1]],
        [[60], [1], [5, 10, 15], [1]],
        [[60], [1], [3], [0]],
        [[40], [1], [2], [0]],
        # [[6], [1], [1], [0]],
        # [[12, 24], [2], [2], [0]],
        # [[18, 36], [3], [3], [0]],
        # [[24, 48], [4], [4], [0]],
    ]

    output_folder = os.path.join(root_path, "framerate_results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_video in input_videos:
        for param in params:
            (
                drop_windows_sizes,
                drop_windows_intervals,
                drop_frame_nums,
                drop_frame_modes,
            ) = param
            for drop_windows_size in drop_windows_sizes:
                for drop_windows_interval in drop_windows_intervals:
                    for drop_frame_num in drop_frame_nums:
                        for drop_frame_mode in drop_frame_modes:
                            print(
                                start_no,
                                frame_no,
                                drop_windows_size,
                                drop_windows_interval,
                                drop_frame_num,
                                drop_frame_mode,
                                input_video,
                            )
                            frame_rate_simulate(
                                start_no,
                                frame_no,
                                drop_windows_size,
                                drop_windows_interval,
                                drop_frame_num,
                                drop_frame_mode,
                                input_video,
                            )
