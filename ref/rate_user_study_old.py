import numpy as np

rank_mapping = {0: 1, 5: 2, 10: 3, 15: 4}

video_postfix_rank = [
    {"60_1_0_0": 1, "60_1_5_1": 2, "60_1_10_1": 3, "60_1_15_1": 4},
    {"60_1_0_0": 1, "15_1_1_0": 2, "15_1_2_0": 3, "15_1_3_0": 4},
    {"60_1_0_0": 1, "60_1_10_1": 2, "12_1_2_0": 3, "60_1_10_2": 4},
]

video_name_postfix = [
    {
        "60_1_0_0": "60FPS",
        "60_1_5_1": "55FPS",
        "60_1_10_1": "50FPS",
        "60_1_15_1": "45FPS",
    },
    {
        "60_1_0_0": "60FPS",
        "15_1_1_0": "Drop1FrameEvery15Frame",
        "15_1_2_0": "Drop2FrameEvery15Frame",
        "15_1_3_0": "Drop3FrameEvery15Frame",
    },
    {
        "60_1_0_0": "60FPS",
        "60_1_10_1": "50FPSEvenlyDrop",
        "12_1_2_0": "50FPSConsecutivelyDrop",
        "60_1_10_2": "50FPSRandomlyDrop",
    },
]


def read_video_identity_file(file_path):
    lines = open(file_path).readlines()
    identities = {}
    for i in range(len(lines) // 4 // 3):
        for k in range(3):
            ranks = []
            video_name = None
            for j in range(4):
                line = lines[i * 12 + k * 4 + j]
                items = line.split(",")

                cur_video_name = items[0][:-6]
                if video_name == None:
                    video_name = cur_video_name
                elif cur_video_name != video_name:
                    raise

                fps = "_".join(items[1].split(".")[0].split("_")[-4:])
                ranks.append(video_postfix_rank[k][fps])
            identities[video_name] = ranks

    return identities


def read_video_names(file_path):
    lines = open(file_path).readlines()
    identities = {}
    for i in range(len(lines) // 4 // 3):
        for k in range(3):
            ranks = []
            video_name = None
            for j in range(4):
                line = lines[i * 12 + k * 4 + j]
                items = line.split(",")

                cur_video_name = items[0][:-6]
                if video_name == None:
                    video_name = cur_video_name
                elif cur_video_name != video_name:
                    raise

                fps = "_".join(items[1].split(".")[0].split("_")[-4:])
                ranks.append(video_name + "_" + video_name_postfix[k][fps])
            identities[video_name] = ranks

    return identities


def read_user_record(header, line):
    headers = header.split(",")
    items = line.split(",")
    time = int(items[3].replace('"', "").strip())
    qq = int(items[12].replace('"', "").strip())
    reflex = int(items[13].replace('"', "").strip())
    age = int(items[15].replace('"', "").strip())
    st_idx = 16

    results = {}
    for i in range(len(items[st_idx:]) // 37):
        game_exp = items[st_idx + i * 37]
        if game_exp.startswith("C"):
            continue

        for j in range(6):
            ranks = []
            video_name = None
            for k in range(4):
                cur_video_name = headers[st_idx + i * 37 + j * 6 + k].split(":")[-1][
                    :-6
                ]
                if video_name == None:
                    video_name = cur_video_name
                elif cur_video_name != video_name:
                    raise
                ranks.append(
                    int(items[st_idx + i * 37 + j * 6 + k + 1].replace('"', "").strip())
                )

            ranks.append(
                int(items[st_idx + i * 37 + j * 6 + 5].replace('"', "").strip())
            )
            ranks.append(
                int(items[st_idx + i * 37 + j * 6 + 6].replace('"', "").strip())
            )
            results[video_name] = ranks

    # return [qq, age, reflex, results]
    return [qq, age, time, reflex, results]


def read_all_user_records(file_path):
    lines = open(file_path, "r", encoding="UTF-8").readlines()
    records = []
    for i in range(1, len(lines)):
        records.append(read_user_record(lines[0], lines[i]))

    return records


def cal_spearman_rank_coef(x, y):
    x = np.array(x)
    y = np.array(y)
    n = x.size

    return 1 - 6 * np.power((x - y), 2).sum() / n / (n**2 - 1)


def cal_user_accuracy(records, identity):
    results = []
    for record in records:
        acc_sum = 0
        conf_sum = 0
        cnt = 0
        for video_name in record[-1].keys():
            acc = cal_spearman_rank_coef(
                identity[video_name], record[-1][video_name][:-2]
            )

            acc_sum += acc
            conf_sum += record[-1][video_name][-1]
            cnt += 1
        if cnt < 12:
            raise
        results.append(record[:-1] + [acc_sum / cnt, conf_sum / cnt])
    return results


def clean_user_record(records, identity):
    results = []
    for record in records:
        cnt = 0
        video_ranks = []
        for video_name in record[-1].keys():
            video_rank = []
            rank = np.argsort(record[-1][video_name][:-2]).tolist()
            for i in rank:
                video_rank.append(identity[video_name][i])

            # video_rank += record[-1][video_name][:-2]
            # video_rank += identity[video_name]
            video_rank.append(record[-1][video_name][-1])
            video_rank.append(identity[video_name][record[-1][video_name][-2] - 1])

            cnt += 1
            video_ranks.append(video_rank)
        if cnt < 12:
            raise
        results.append(record[:-1] + video_ranks)
    return results


def cal_video_accuracy(records, identity):
    results = []

    for video_name in identity.keys():
        acc_sum = 0
        conf_sum = 0
        cnt = 0

        for record in records:
            if record[2] < 1000:
                continue
            if video_name not in record[-1]:
                continue

            acc = cal_spearman_rank_coef(
                identity[video_name], record[-1][video_name][:-2]
            )

            acc_sum += acc
            conf_sum += record[-1][video_name][-1]
            cnt += 1
            # if(video_name == 'dnf_2'):
            #     print(video_name, acc)
        if cnt > 0:
            results.append([video_name] + [cnt, acc_sum / cnt, conf_sum / cnt])
    return results


if __name__ == "__main__":
    video_names = read_video_names("test_data/game_video_identity.csv")
    records = read_all_user_records("test_data/13425417_202311301628374753.csv")
    cleaned_records = clean_user_record(records, video_names)

    identity = read_video_identity_file("test_data/game_video_identity.csv")
    records = read_all_user_records("test_data/13425417_202311301628374753.csv")
    user_accs = cal_user_accuracy(records, identity)
    video_accs = cal_video_accuracy(records, identity)

    res_file = open("test_data/user_study_result.csv", "w")
    res_file.write("QQ, age, time, reflex, accuracy, confidence\n")
    for line in user_accs:
        res_file.write("{}, {}, {}, {}, {:.2f}, {:.2f}\n".format(*line))

    res_file.write("video, cnt, accuracy, confidence\n")
    for line in video_accs:
        res_file.write("{}, {}, {:.2f}, {:.2f}\n".format(*line))

    res_file = open("test_data/cleaned_records.csv", "w")
    res_file.write("QQ, age, time, reflex, accuracy, confidence\n")
    for i in range(len(cleaned_records)):
        res_file.write("{}, {}, {}, {}, {:.2f}, {:.2f}\n".format(*user_accs[i]))
        for rank in cleaned_records[i][4:]:
            res_file.write("{}, {}, {}, {}, {}, {}\n".format(*rank))
        res_file.write("\n")
    exit()
