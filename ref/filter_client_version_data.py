import os, sys, shutil


def get_pari(log_path):
    lines = open(log_path).readlines()
    id_sdk_pair = {}
    id_f11_pair = {}
    for line in lines:
        items = line.strip().split()
        sid = items[4][4:]
        sdk = items[283].split(";")[1]
        id_sdk_pair[sid] = sdk
        id_f11_pair[sid] = items[236].split(",")[1]
    return id_sdk_pair, id_f11_pair


def process_all_data(root_path, sdk_versions=[]):
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for session_folder in os.listdir(data_path):
            if not session_folder.startswith("session_info"):
                continue

            session_path = os.path.join(data_path, session_folder)
            summary_path = os.path.join(
                data_path, "summary_info_" + session_folder[13:] + "_raw.log"
            )

            sid_sdk_pair, sid_f11_pair = get_pari(summary_path)

            for log_name in os.listdir(session_path):
                if not log_name.endswith(".csv"):
                    continue

                sid = log_name.split("_")[1]
                file_path = os.path.join(session_path, log_name)

                # print(file_path, sid, sid_sdk_pair[sid])
                if sid_sdk_pair[sid] not in sdk_versions:
                    target_path = os.path.join(os.path.dirname(file_path), "wrong_sdk")
                    if not os.path.exists(target_path):
                        os.makedirs(target_path, exist_ok=True)
                    file_name = os.path.basename(file_path)
                    shutil.move(file_path, os.path.join(target_path, file_name))
                    print(
                        "move file: %s sdk: %s to %s"
                        % (
                            file_path,
                            sid_sdk_pair[sid],
                            os.path.join(target_path, file_name),
                        )
                    )

                elif sid_f11_pair[sid].startswith("F11=0"):
                    target_path = os.path.join(os.path.dirname(file_path), "not_vsync")
                    if not os.path.exists(target_path):
                        os.makedirs(target_path, exist_ok=True)
                    file_name = os.path.basename(file_path)
                    shutil.move(file_path, os.path.join(target_path, file_name))
                    print(
                        "move file: %s f11: %s to %s"
                        % (
                            file_path,
                            sid_f11_pair[sid],
                            os.path.join(target_path, file_name),
                        )
                    )

                elif sid_f11_pair[sid].startswith("F11=00") or sid_f11_pair[
                    sid
                ].startswith("F11=10"):
                    target_path = os.path.join(
                        os.path.dirname(file_path), "not_fullscreen"
                    )
                    if not os.path.exists(target_path):
                        os.makedirs(target_path, exist_ok=True)
                    file_name = os.path.basename(file_path)
                    shutil.move(file_path, os.path.join(target_path, file_name))
                    print(
                        "move file: %s f11: %s to %s"
                        % (
                            file_path,
                            sid_f11_pair[sid],
                            os.path.join(target_path, file_name),
                        )
                    )

                elif not sid_f11_pair[sid].endswith("1"):
                    target_path = os.path.join(
                        os.path.dirname(file_path), "not_fullscreen"
                    )
                    if not os.path.exists(target_path):
                        os.makedirs(target_path, exist_ok=True)
                    file_name = os.path.basename(file_path)
                    shutil.move(file_path, os.path.join(target_path, file_name))
                    print(
                        "move file: %s f11: %s to %s"
                        % (
                            file_path,
                            sid_f11_pair[sid],
                            os.path.join(target_path, file_name),
                        )
                    )


if __name__ == "__main__":
    process_all_data(
        sys.argv[1],
        [
            "cgplayer_sdk:0.10.0.168",
            "cgplayer_sdk:0.10.0.169",
            "cgplayer_sdk:0.10.0.170",
            "cgplayer_sdk:0.10.0.171",
            "cgplayer_sdk:0.10.0.172",
            "cgplayer_sdk:0.10.0.173",
            "cgplayer_sdk:0.10.0.174",
        ],
    )
