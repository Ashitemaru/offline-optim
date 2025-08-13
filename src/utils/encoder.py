import os
import csv
from dataclasses import dataclass
import random
import numpy as np

ENCODER = "D:/Ashitemaru/CodingFolder/Projects/offline-optim/bin/framerate_simulator/demo.exe"
ENCODER_ROOT = "D:/Ashitemaru/CodingFolder/Projects/offline-optim/bin/framerate_simulator"

@dataclass
class EncoderConfig:
    drop_window_size: int
    drop_window_interval: int
    drop_frame_num: int
    drop_frame_mode: int

def encode(config: EncoderConfig, input_path, output_root, drop_start_idx=60, total_frame_cnt=900):
    assert input_path.endswith(".yuv")
    output_path = os.path.join(
        output_root,
        "encoded_video",
        "%s_winsize%d_winint%d_dropnum%d_mode%d.264"
        % (
            os.path.basename(input_path)[:-4],
            config.drop_window_size,
            config.drop_window_interval,
            config.drop_frame_num,
            config.drop_frame_mode,
        ),
    )
    encoder_log_path = os.path.join(output_root, "encoder.log")
    if os.path.exists(output_path):
        return
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    file_path = os.path.join(ENCODER_ROOT, "frame_flag.csv")
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(drop_start_idx):
            writer.writerow([i, 1])

        writer.writerow([drop_start_idx, 1])
        windows_num = np.ceil((total_frame_cnt - drop_start_idx) / config.drop_window_size).astype(int)
        for i in range(windows_num):
            if i % config.drop_window_interval == 0:
                n = config.drop_frame_num
                for j in range(i * config.drop_window_size, (i + 1) * config.drop_window_size):
                    if j == (total_frame_cnt - drop_start_idx) - 1:
                        continue
                    if config.drop_frame_mode == 0:  # continous mode
                        if n > 0:
                            writer.writerow([j + 1 + drop_start_idx, 0])
                            n -= 1
                        else:
                            writer.writerow([j + 1 + drop_start_idx, 1])
                    if config.drop_frame_mode == 1:  # homogeneous mode
                        inter_interval = config.drop_window_size // config.drop_frame_num
                        if (j - i * config.drop_window_size) % inter_interval == 0 and n > 0:
                            n -= 1
                            writer.writerow([j + 1 + drop_start_idx, 0])
                        else:
                            writer.writerow([j + 1 + drop_start_idx, 1])
                    if config.drop_frame_mode == 2:  # random mode
                        random_array = np.array(random.sample(range(0, config.drop_window_size - 1), config.drop_frame_num))
                        if (j - i * config.drop_window_size) in random_array:
                            writer.writerow([j + 1 + drop_start_idx, 0])
                        else:
                            writer.writerow([j + 1 + drop_start_idx, 1])
            else:
                for j in range(i * config.drop_window_size, (i + 1) * config.drop_window_size):
                    if j == (total_frame_cnt - drop_start_idx) - 1:
                        continue
                    writer.writerow([j + 1 + drop_start_idx, 1])

    bitrate = random.sample(range(35, 40), 1)[0]
    cmd = "{} -input_seq {} -output_bin {} -color_space 3 -width 1920 -height 1080 -framerate 60 -bitrate {} -max_bitrate {} -encoder_type 0 -codec_type 16 -gpu_index 0 -gop_size 60 -ref_frames 3 -black_frame_detect 0 -business_id 1 -data_analysis 1 -vmaf 0 -vmaf_thread_num 0 > {}".format(
        ENCODER, input_path, output_path, bitrate * 1000, bitrate * 1000, encoder_log_path
    )
    # print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    encode(
        config=EncoderConfig(
            drop_window_size=60,
            drop_window_interval=1,
            drop_frame_num=5,
            drop_frame_mode=0,
        ),
        input_path="C:/Users/ashit/Downloads/game_starrail_file_0_slice_7_type_skill.yuv",
        output_root="../output",
    )