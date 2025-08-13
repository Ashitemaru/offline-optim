import os, shutil, random

# video_group_postfix1 = [
#     [
#         '60_1_0_0',
#         '60_1_5_1',
#         '60_1_10_1',
#         '60_1_15_1'
#     ],
#     [
#         '60_1_0_0',
#         '15_1_1_0',
#         '15_2_2_0',
#         '15_3_3_0'
#     ],
#     [
#         '60_1_0_0',
#         '60_1_10_1',
#         '60_1_10_2',
#         '12_1_2_0'
#     ]
# ]

video_group_postfix1 = [
    ["60_1_0_0", "60_1_5_1", "60_1_10_1", "60_1_15_1"],  # 60  # 55  # 50  # 45
    [
        "60_1_0_0",  # 60
        "60_1_1_1",  # 59
        "60_1_2_1",  # 58
        "60_1_3_1",  # 57
    ],
    # [
    #     '6_1_1_0',  #50
    #     '12_2_2_0', #50, drop 2 frames consecutively
    #     '18_3_3_0', #50, drop 3 frames consecutively
    #     '24_4_4_0', #50, drop 4 frames consecutively
    # ],
    # [
    #     '60_1_5_1', #55
    #     '24_2_2_0', #55, drop 2 frames consecutively
    #     '36_3_3_0', #55, drop 3 frames consecutively
    #     '48_4_4_0', #55, drop 4 frames consecutively
    # ],
    [
        "60_1_0_0",  # 60
        "60_1_3_1",  # 57, drop 1 frames consecutively
        "40_1_2_0",  # 57, drop 2 frames consecutively
        "60_1_3_0",  # 57, drop 3 frames consecutively
    ],
    # [
    #     '60_1_0_0', #60
    #     '60_1_1_1', #59
    # ],
]

video_group_postfix2 = [
    [
        "60_1_0_0_0.00_0.00",
        "60_1_60_1_0.00_0.25",
        "60_1_60_1_0.25_0.75",
        "60_1_60_1_0.75_1.00",
    ],  # different positions, all frame tearing
    [
        "60_1_1_1_0.25_0.75",
        "60_1_3_1_0.25_0.75",
        "60_1_6_1_0.25_0.75",
        "60_1_15_1_0.25_0.75",
    ],  # center region, different No. of frames
    [
        "60_1_6_0_0.25_0.75",
        "60_1_6_1_0.25_0.75",
        "60_1_15_0_0.25_0.75",
        "60_1_15_1_0.25_0.75",
    ],  # center region, consecutive vs uniformly
    [
        "60_1_6_0_0.25_0.75",
        "60_3_6_0_0.25_0.75",
        "60_5_6_0_0.25_0.75",
        "60_5_15_0_0.25_0.75",
        # '60_3_15_0_0.25_0.75',
    ],  # center region, different window interval
    [
        "60_1_1_1_0.00_0.25",
        "60_1_3_1_0.00_0.25",
        "60_1_6_1_0.00_0.25",
        "60_1_15_1_0.00_0.25",
    ],  # top region, different No. of frames
    [
        "60_1_6_0_0.00_0.25",
        "60_1_6_1_0.00_0.25",
        "60_1_15_0_0.00_0.25",
        "60_1_15_1_0.00_0.25",
    ],  # top region, consecutive vs uniformly
    [
        "60_1_6_0_0.00_0.25",
        "60_3_6_0_0.00_0.25",
        "60_5_6_0_0.00_0.25",
        "60_5_15_0_0.00_0.25",
        # '60_3_15_0_0.00_0.25',
    ],  # top region, different window interval
]

video_names = [
    # 'lostark_1920_1080_chunk_4_900',
    # 'lostark_1920_1080_chunk_5_900',
    # 'valorant_1920_1080_chunk_11_900',
    # 'valorant_1920_1080_chunk_24_900',
    # 'lol_1920_1080_chunk_1_900',
    # 'lol_1920_1080_chunk_2_900',
    # 'yuanshen_1920_1080_chunk_4_900',
    # 'yuanshen_1920_1080_chunk_10_900',
    # 'dnf_1920_1080_chunk_1_900',
    # 'dnf_1920_1080_chunk_2_900',
    # 'nba2k_1920_1080_chunk_1_900',
    # 'nba2k_1920_1080_chunk_3_900',
    "nizhan_1920_1080_chunk_13_900",
    "nizhan_1920_1080_chunk_15_900",
    "starrail_1920_1080_chunk_8_900",
    "starrail_1920_1080_chunk_10_900",
]


def convert_video_group(input_dir, output_dir, video_group_postfix):
    video_idx = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = open(os.path.join(output_dir, "game_video_identity.csv"), "a")

    for video_full_name in video_names:
        game_name = video_full_name.split("_")[0]
        video_root_path = os.path.join(output_dir, game_name)
        if not os.path.exists(video_root_path):
            os.makedirs(video_root_path)

        for group in video_group_postfix:
            if game_name not in video_idx:
                cur_idx = 1
                video_idx[game_name] = 1
            else:
                video_idx[game_name] += 1
                cur_idx = video_idx[game_name]

            random.shuffle(group)
            print(group)
            for i, postfix in enumerate(group):
                cur_input_video_name = video_full_name + "_" + postfix + ".mp4"
                cur_output_video_name = "%s_%d_%d.mp4" % (game_name, cur_idx, i + 1)
                video_path = os.path.join(input_dir, cur_input_video_name)
                dst_path = os.path.join(video_root_path, cur_output_video_name)

                print(video_path, dst_path)
                shutil.copyfile(video_path, dst_path)
                log_file.write(
                    "%s,%s\n" % (cur_output_video_name, cur_input_video_name)
                )


def convert_all(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = open(os.path.join(output_dir, "video_identity.csv"), "w")

    used_no = {}
    game_cnt = {}
    game_idx = {}
    for video in os.listdir(input_dir):
        basename = os.path.basename(video)[:-4]
        header = "_".join(basename.split("_")[:5])
        game_name = basename.split("_")[0]

        if header in game_idx:
            cur_idx = game_idx[header]
        elif game_name not in game_cnt:
            cur_idx = 1
            game_cnt[game_name] = 1
            game_idx[header] = 1
        else:
            cur_idx = game_cnt[game_name] + 1
            game_cnt[game_name] = cur_idx
            game_idx[header] = cur_idx

        selected_no = 0
        if header not in used_no:
            candidate_no = list(range(1, 5))
            selected_no = random.sample(candidate_no, 1)[0]
            candidate_no.remove(selected_no)
            used_no[header] = candidate_no
        else:
            selected_no = random.sample(used_no[header], 1)[0]
            used_no[header].remove(selected_no)

        video_path = os.path.join(input_dir, video)
        dst_path = os.path.join(
            output_dir, "%s_%d_%d.mp4" % (game_name, cur_idx, selected_no)
        )

        shutil.copyfile(video_path, dst_path)

        log_file.write("%s,%s\n" % (dst_path, video_path))


if __name__ == "__main__":
    convert_video_group(
        r"D:\e2e_videos\framerate_results_converted",
        r"D:\e2e_videos\framerate_results_randomized",
        video_group_postfix1,
    )
    # convert_video_group(r"D:\e2e_videos\tearing_results_converted", r"D:\e2e_videos\tearing_results_randomized", video_group_postfix2)
