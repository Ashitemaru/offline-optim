import os, sys, shutil
import numpy as np
import pandas as pd


# file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,max_fps,min_fps,origin_fps,optimized_fps-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,optimized_objective_fps-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,optimized_noloss_fps-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,origin_render_queue,optimized_render_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,extra_display_ts-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,optimized_total_render_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,avg_dec_time,avg_dec_tot_time,avg_render_time,avg_proc_time,tot_frame_no,tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,network_i_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,display_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,netts_over100_cnt,renders_over12_cnt,exploitable_network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_network_i_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_display_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo,exploitable_near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo
def get_original_path(log_path):
    return log_path[:-11] + ",".join(log_path[-11:].split("_"))


def find_problem_flow(log_path, log_no=20):
    df = pd.read_csv(log_path)

    stall_thr = 0.001
    min_fps_thr = 55
    max_fps_thr = 65

    valid_idx = np.where(
        np.logical_and.reduce(
            (
                df["avg_render_time"].to_numpy() <= 10,
                df["max_fps"].to_numpy() >= min_fps_thr,
                df["max_fps"].to_numpy() <= max_fps_thr,
                df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
            )
        )
    )[0]
    new_df = df.iloc[valid_idx, :]

    all_lines = open(log_path).readlines()[1:]
    lines = [all_lines[idx] for idx in valid_idx]

    # 'tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'display_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
    # 'near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo'

    file_paths = new_df["file_name"].to_list()
    big_frame_ratio = new_df[
        "network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    dl_jitter_ratio = new_df[
        "network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    stall_ratio = new_df[
        "network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    packet_loss_ratio = new_df[
        "network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    render_jitter_ratio = new_df[
        "render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    server_jitter_ratio = new_df[
        "server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    decoder_jitter_ratio = new_df[
        "decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    near_vsync_jitter_ratio = new_df[
        "near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ] / np.maximum(
        new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"], 1
    )
    header = ",".join(new_df.columns)

    top_idx = np.argsort(-np.array(big_frame_ratio))[:log_no]
    target_path = os.path.join("test_data", "big_frame_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(dl_jitter_ratio))[:log_no]
    target_path = os.path.join("test_data", "dl_jitter_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(stall_ratio))[:log_no]
    target_path = os.path.join("test_data", "stall_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(packet_loss_ratio))[:log_no]
    target_path = os.path.join("test_data", "packet_loss_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(render_jitter_ratio))[:log_no]
    target_path = os.path.join("test_data", "render_jitter_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(decoder_jitter_ratio))[:log_no]
    target_path = os.path.join("test_data", "decoder_jitter_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(server_jitter_ratio))[:log_no]
    target_path = os.path.join("test_data", "server_jitter_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()

    top_idx = np.argsort(-np.array(near_vsync_jitter_ratio))[:log_no]
    target_path = os.path.join("test_data", "near_vsync_jitter_ratio")
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    output_file = open(os.path.join(target_path, "result.csv"), "a")
    output_file.write(header)
    for idx in top_idx:
        cur_file_path = get_original_path(file_paths[idx])
        shutil.copy(
            cur_file_path,
            os.path.join(target_path, "_".join(cur_file_path.split(os.sep)[4:])),
        )
        output_file.write(lines[idx])
    output_file.close()


if __name__ == "__main__":
    find_problem_flow(sys.argv[1])
