import sys


def sim(file_path):
    lines = open(file_path).readlines()

    out_file = open(file_path[:-4] + "_with_flag.log", "w")
    cnt = 0

    for line in lines:
        items = line.strip().split()
        avg_render_time = float(items[18])
        frame_interval = int(items[20])
        vsync_slot_threshold = int(items[22])

        prev_frame_ready_ts = int(items[28])
        prev_expected_display_ts = int(items[30])
        prev_actual_display_ts = int(items[32])
        prev_display_slot = int(items[34])
        cur_frame_ready_ts = int(items[46])
        cur_expected_display_ts = int(items[50])
        cur_actual_display_ts = int(items[54])
        cur_display_slot = int(items[60])

        slot_delayed = int(items[64])
        slot_moved = int(items[66])

        flag = " 0"
        # if prev_display_slot == cur_display_slot and \
        #     (prev_frame_ready_ts > cur_expected_display_ts or prev_expected_display_ts == cur_expected_display_ts) and \
        #     cur_frame_ready_ts < cur_expected_display_ts + frame_interval:
        #     if slot_delayed <= 0 or prev_actual_display_ts < cur_expected_display_ts:
        #         flag = ' 1'
        #         cnt += 1

        if (
            cur_display_slot == prev_display_slot + 1
            and cur_frame_ready_ts < cur_expected_display_ts
            and cur_actual_display_ts > cur_expected_display_ts
            and (
                slot_moved == 0
                or prev_frame_ready_ts >= cur_expected_display_ts - frame_interval
            )
        ):
            flag = " 1"
            cnt += 1

        out_file.write(line.strip() + flag + "\n")
    print(cnt, len(lines), cnt / len(lines) * 60)


if __name__ == "__main__":
    sim(sys.argv[1])
