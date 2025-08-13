import os, shutil


def cal_netts_log(root_path):
    output_file_path = open("test_data/trace_fps_summary.csv", "w")
    output_file_path.write(
        "path, IP, sid, connect_time, cgs_fps, proxy_fps, client_fps, proxy_valid_fps, client_recv_valid_fps, client_decode_valid_fps, client_invoke_valid_fps, client_actual_valid_fps\n"
    )

    cgs_jitter_path = os.path.join("test_data", "frame_jitter_log", "cgs_jitter")
    if not os.path.exists(cgs_jitter_path):
        os.makedirs(cgs_jitter_path)
    cgs_jitter_output_path = open(
        os.path.join(cgs_jitter_path, "trace_fps_summary.csv"), "w"
    )
    cgs_jitter_output_path.write(
        "path, IP, sid, connect_time, cgs_fps, proxy_fps, client_fps, proxy_valid_fps, client_recv_valid_fps, client_decode_valid_fps, client_invoke_valid_fps, client_actual_valid_fps\n"
    )

    net_jitter_path = os.path.join("test_data", "frame_jitter_log", "net_jitter")
    if not os.path.exists(net_jitter_path):
        os.makedirs(net_jitter_path)
    net_jitter_output_path = open(
        os.path.join(net_jitter_path, "trace_fps_summary.csv"), "w"
    )
    net_jitter_output_path.write(
        "path, IP, sid, connect_time, cgs_fps, proxy_fps, client_fps, proxy_valid_fps, client_recv_valid_fps, client_decode_valid_fps, client_invoke_valid_fps, client_actual_valid_fps\n"
    )

    cleaned_data = set()
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)
        log_path = os.path.join(data_path, "netts.log")

        if not os.path.exists(log_path):
            continue

        for line in open(log_path).readlines():
            items = line.strip().split()
            ip = items[230]
            sid = items[4]
            conn_time = int(items[103])
            date = items[0].replace("/", "-")

            if conn_time == 0:
                print(data_path, ip, conn_time)
                continue

            info = (
                int(items[6]),
                int(items[10]),
                int(items[8]),
                int(items[12]),
                int(items[348]),
                int(items[350]),
                int(items[352]),
                int(items[354]),
                int(items[356]),
            )

            file_name = "%s_%s_%s_%d,%d,%d,%d,%d,%d,%d,%d,%d.csv" % (
                ip,
                date,
                sid.split(":")[1],
                *info,
            )
            trace_path = os.path.join(data_path, file_name)
            if not os.path.exists(trace_path):
                # print(trace_path)
                continue

            if trace_path in cleaned_data:
                # print('multiple occurance:', trace_path)
                continue

            cleaned_data.add(trace_path)

            frame_total = int(items[33])
            frame_ack = int(items[35])
            pause_cnt = int(items[152])
            cgs_frame_valid = int(items[358])
            client_recv_valid = int(items[360])
            client_decode_valid = int(items[362])
            client_invoke_valid = int(items[364])
            client_display_valid = int(items[368])
            if (
                frame_total < 3000
                or frame_total / conn_time > 75
                or frame_total / conn_time < 40
            ):
                continue

            output_file_path.write(
                "%s,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                % (
                    data_path,
                    ip,
                    sid,
                    conn_time,
                    (frame_total + pause_cnt) / conn_time,
                    frame_total / conn_time,
                    frame_ack / conn_time,
                    cgs_frame_valid / conn_time,
                    client_recv_valid / conn_time,
                    client_decode_valid / conn_time,
                    client_invoke_valid / conn_time,
                    client_display_valid / conn_time,
                )
            )

            if (
                (frame_total + pause_cnt) / conn_time < 65
                and pause_cnt / conn_time < 0.5
                and (frame_ack - cgs_frame_valid) / conn_time > 5
            ):
                shutil.copy(trace_path, os.path.join(cgs_jitter_path, file_name))
                cgs_jitter_output_path.write(
                    "%s,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (
                        data_path,
                        ip,
                        sid,
                        conn_time,
                        (frame_total + pause_cnt) / conn_time,
                        frame_total / conn_time,
                        frame_ack / conn_time,
                        cgs_frame_valid / conn_time,
                        client_recv_valid / conn_time,
                        client_decode_valid / conn_time,
                        client_invoke_valid / conn_time,
                        client_display_valid / conn_time,
                    )
                )
                print(
                    "cgs_jitter %s,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (
                        data_path,
                        ip,
                        sid,
                        conn_time,
                        (frame_total + pause_cnt) / conn_time,
                        frame_total / conn_time,
                        frame_ack / conn_time,
                        cgs_frame_valid / conn_time,
                        client_recv_valid / conn_time,
                        client_decode_valid / conn_time,
                        client_invoke_valid / conn_time,
                        client_display_valid / conn_time,
                    )
                )

            if (
                (frame_total + pause_cnt) / conn_time < 65
                and (frame_ack - cgs_frame_valid) / conn_time < 1
                and (cgs_frame_valid - client_recv_valid) / conn_time > 5
            ):
                shutil.copy(trace_path, os.path.join(net_jitter_path, file_name))
                net_jitter_output_path.write(
                    "%s,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (
                        data_path,
                        ip,
                        sid,
                        conn_time,
                        (frame_total + pause_cnt) / conn_time,
                        frame_total / conn_time,
                        frame_ack / conn_time,
                        cgs_frame_valid / conn_time,
                        client_recv_valid / conn_time,
                        client_decode_valid / conn_time,
                        client_invoke_valid / conn_time,
                        client_display_valid / conn_time,
                    )
                )
                print(
                    "net_jitter %s,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
                    % (
                        data_path,
                        ip,
                        sid,
                        conn_time,
                        (frame_total + pause_cnt) / conn_time,
                        frame_total / conn_time,
                        frame_ack / conn_time,
                        cgs_frame_valid / conn_time,
                        client_recv_valid / conn_time,
                        client_decode_valid / conn_time,
                        client_invoke_valid / conn_time,
                        client_display_valid / conn_time,
                    )
                )


if __name__ == "__main__":
    cal_netts_log("/data/home/clwwwu/client_version_data")
