import os, sys, shutil
import numpy as np


def get_original_path(log_path):
    return log_path[:-11] + ",".join(log_path[-11:].split("_"))


def find_potential_gain_flow(
    log_path, sub_folder="sample", root_folder="/mydata/clwwwu/frame_log/"
):
    lines = open(log_path).readlines()[1:]

    target_path = os.path.join(root_folder, sub_folder)
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "unit_test.txt"), "w")
    for line in lines:
        items = line.strip().split(",")

        cur_file_path = get_original_path(items[0])
        new_file_name = "_".join(cur_file_path.split(os.sep)[4:])
        shutil.copy(cur_file_path, os.path.join(target_path, new_file_name))
        output_file.write(new_file_name + "\n")

    output_file.close()


if __name__ == "__main__":
    # find_potential_gain_flow("test_data/result-potential_gain2.csv", sub_folder='sample_gain2')
    find_potential_gain_flow(
        "test_data/result-potential_gain3.5.csv", sub_folder="sample_gain3.5"
    )
    # find_potential_gain_flow("test_data/result-potential_gain15.0.csv", sub_folder='sample_gain15')
