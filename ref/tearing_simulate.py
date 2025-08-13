import os
import random
import numpy as np
import csv

root_path = "D:\e2e_videos"


def tearing_simulate(
    start_no,
    frame_no,
    tearing_windows_size,
    tearing_windows_interval,
    tearing_frame_num,
    tearing_frame_mode,
    tearing_range,
    input_video,
):
    if tearing_frame_num > tearing_windows_size:
        print("drop num is larger than drop windows size!")
        return

    print("input_video:", input_video)
    print("\ttearing_windows_size:", tearing_windows_size)
    print("\ttearing_windows_interval:", tearing_windows_interval)
    print("\ttearing_frame_num:", tearing_frame_num)
    print("\ttearing_frame_mode:", tearing_frame_mode)
    print("\ttearing_range: %.2f %.2f" % (tearing_range[0], tearing_range[1]))

    output_path = os.path.join(
        root_path,
        "tearing_results",
        "%s_%d_%d_%d_%d_%d_%.2f_%.2f.264"
        % (
            os.path.basename(input_video)[:-4],
            start_no + frame_no,
            tearing_windows_size,
            tearing_windows_interval,
            tearing_frame_num,
            tearing_frame_mode,
            *tearing_range,
        ),
    )

    if os.path.exists(output_path):
        return

    # Open the CSV file in write mode
    file_path = os.path.join(root_path, "tearing_simulator", "frame_flag.csv")
    with open(file_path, "w", newline="") as f:
        # Create a CSV writer object
        writer = csv.writer(f)
        for i in range(start_no):
            writer.writerow([i, 1])

        writer.writerow([start_no, 1])
        windows_num = np.ceil(frame_no / tearing_windows_size).astype(int)
        for i in range(windows_num):
            if i % tearing_windows_interval == 0:  ## windows_interval
                n = tearing_frame_num
                L1 = random.sample(range(0, tearing_windows_size), tearing_frame_num)
                # L1 = random.sample(range(0,tearing_windows_size-1), tearing_frame_num)
                random_array = np.array(L1)
                for j in range(
                    i * tearing_windows_size, (i + 1) * tearing_windows_size
                ):
                    if j == frame_no - 1:
                        continue
                    if tearing_frame_mode == 0:  ## continous mode
                        if n > 0:
                            writer.writerow(
                                [j + 1 + start_no, random.uniform(*tearing_range) % 1]
                            )
                            n -= 1
                        else:
                            writer.writerow([j + 1 + start_no, 1])
                    if tearing_frame_mode == 1:  ## homogeneous mode
                        inter_interval = tearing_windows_size // tearing_frame_num
                        if (
                            j - i * tearing_windows_size
                        ) % inter_interval == 0 and n > 0:
                            n -= 1
                            writer.writerow(
                                [j + 1 + start_no, random.uniform(*tearing_range) % 1]
                            )
                        else:
                            writer.writerow([j + 1 + start_no, 1])
                    if tearing_frame_mode == 2:  ## random mode
                        if (j - i * tearing_windows_size) in random_array:
                            writer.writerow(
                                [j + 1 + start_no, random.uniform(*tearing_range) % 1]
                            )
                        else:
                            writer.writerow([j + 1 + start_no, 1])
            else:
                for j in range(
                    i * tearing_windows_size, (i + 1) * tearing_windows_size
                ):
                    if j == frame_no - 1:
                        continue
                    writer.writerow([j + 1 + start_no, 1])

    cmd = "D:\\e2e_videos\\tearing_simulator\\demo.exe -input_seq {} -output_bin {} -color_space 3 -width 1920 -height 1080 -framerate 60 -bitrate 50000 -max_bitrate 50000 -encoder_type 0 -codec_type 16 -gpu_index 0 -gop_size 60 -ref_frames 3 -black_frame_detect 0 -business_id 1 -data_analysis 1 -vmaf 0 -vmaf_thread_num 0 > D:\\e2e_videos\\tearing_simulator\\anchor_264.log".format(
        input_video, output_path
    )
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    input_videos = [
        # r"D:\e2e_videos\Yuanshen\yuv\Yuanshen_1920_1080_chunk_4.yuv",
        # r"D:\e2e_videos\Yuanshen\yuv\Yuanshen_1920_1080_chunk_10.yuv",
        # r"D:\e2e_videos\Nizhan\yuv\Nizhan_1920_1080_chunk_13.yuv",
        # r"D:\e2e_videos\Nizhan\yuv\Nizhan_1920_1080_chunk_15.yuv",
        # r"D:\e2e_videos\StartRail\yuv\StarRail_1920_1080_chunk_8.yuv",
        # r"D:\e2e_videos\StartRail\yuv\StarRail_1920_1080_chunk_10.yuv",
        r"D:\e2e_videos\Valorant\yuv\Valorant_1920_1080_chunk_11.yuv",
        r"D:\e2e_videos\Valorant\yuv\Valorant_1920_1080_chunk_20.yuv",
        r"D:\e2e_videos\Valorant\yuv\Valorant_1920_1080_chunk_24.yuv",
        # r"D:\e2e_videos\Others\yuv\DNF_1920_1080_chunk_1.yuv",
        # r"D:\e2e_videos\Others\yuv\DNF_1920_1080_chunk_2.yuv",
        # r"D:\e2e_videos\Others\yuv\DNF_1920_1080_chunk_3.yuv",
        # r"D:\e2e_videos\Others\yuv\LoL_1920_1080_chunk_1.yuv",
        # r"D:\e2e_videos\Others\yuv\LoL_1920_1080_chunk_2.yuv",
        # r"D:\e2e_videos\Others\yuv\LoL_1920_1080_chunk_3.yuv",
        r"D:\e2e_videos\Others\yuv\LostArk_1920_1080_chunk_3.yuv",
        r"D:\e2e_videos\Others\yuv\LostArk_1920_1080_chunk_4.yuv",
        r"D:\e2e_videos\Others\yuv\LostArk_1920_1080_chunk_5.yuv",
        # r"D:\e2e_videos\Others\yuv\NBA2K_1920_1080_chunk_1.yuv",
        # r"D:\e2e_videos\Others\yuv\NBA2K_1920_1080_chunk_2.yuv",
        # r"D:\e2e_videos\Others\yuv\NBA2K_1920_1080_chunk_3.yuv"
        # r"D:\e2e_videos\Others\yuv\lol_1920_1080_chunk_4.yuv",
        # r"D:\e2e_videos\Others\yuv\lostark_1920_1080_chunk_2.yuv",
        # r"D:\e2e_videos\Others\yuv\lostark_1920_1080_chunk_9.yuv",
        # r"D:\e2e_videos\Others\yuv\nba2k_1920_1080_chunk_3.yuv"
        # r"D:\e2e_videos\Others\yuv\DevilMayCry5_1920x1080_60fps.yuv",
        # r"D:\e2e_videos\Others\yuv\KOFXIV_1920x1080_60fps.yuv"
    ]

    # 5 dimension
    start_no = 60
    frame_no = 900 - start_no
    # tearing_windows_sizes = [15]     # 18 frames = 300ms
    # tearing_windows_intervals = [1]  # windows interval
    # tearing_frame_nums = [15]       # drop frame num in one windows
    # tearing_frame_modes = [0]         # drop mode: 0~continous  1~homogeneous  2~random
    # tearing_ranges = [[0, 0.2], [0, 0.25], [0.75, 1], [0.8, 1]]

    params = [
        # [[60], [1], [0], [0], [(0, 0)]],
        # [[60], [1], [3, 6, 15], [0, 1], [(0, 0.25), (0.25, 0.75), (0.75, 1)]],
        # [[60], [1], [1, 60], [1], [(0, 0.25), (0.25, 0.75), (0.75, 1)]],
        # [[60], [3, 5], [6, 15], [0, 1], [(0, 0.25), (0.25, 0.75), (0.75, 1)]],
        [[60], [1], [60], [0], [(0.5, 0.5)]],
    ]

    output_folder = os.path.join(root_path, "tearing_results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_video in input_videos:
        # tearing_simulate(start_no, frame_no, 60, 1, 0, 0, [1, 1], input_video)
        for param in params:
            (
                tearing_windows_sizes,
                tearing_windows_intervals,
                tearing_frame_nums,
                tearing_frame_modes,
                tearing_ranges,
            ) = param
            for tearing_windows_size in tearing_windows_sizes:
                for tearing_windows_interval in tearing_windows_intervals:
                    for tearing_frame_num in tearing_frame_nums:
                        for tearing_frame_mode in tearing_frame_modes:
                            for tearing_range in tearing_ranges:
                                tearing_simulate(
                                    start_no,
                                    frame_no,
                                    tearing_windows_size,
                                    tearing_windows_interval,
                                    tearing_frame_num,
                                    tearing_frame_mode,
                                    tearing_range,
                                    input_video,
                                )
