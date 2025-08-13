import os, shutil


def convert_all(input_dir, output_dir):
    log_file = os.path.join(output_dir, "game_video_identity.csv")
    assert os.path.isfile(log_file), "No identity file found: %s" % log_file

    for line in open(log_file).readlines():
        items = line.strip().split(",")
        output_file_name = items[0]
        input_file_name = items[1]

        game_name = input_file_name.split("_")[0]

        game_dir = os.path.join(output_dir, game_name)
        if not os.path.exists(game_dir):
            os.makedirs(game_dir)

        src_path = os.path.join(input_dir, input_file_name)
        dst_path = os.path.join(game_dir, output_file_name)

        print(src_path, dst_path)
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    convert_all(
        r"D:\e2e_videos\framerate_results_converted",
        r"D:\e2e_videos\framerate_results_randomized",
    )
