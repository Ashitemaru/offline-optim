import numpy as np
import pandas as pd
import collections
import json

tearing_video_postfix_to_rank = [
    {
        "60_1_0_0_0.00_0.00": 1,
        "60_1_60_1_0.00_0.25": 2,
        "60_1_60_1_0.25_0.75": 4,
        "60_1_60_1_0.75_1.00": 3,
    },  # different positions, all frame tearing
    {
        "60_1_1_1_0.25_0.75": 1,
        "60_1_3_1_0.25_0.75": 2,
        "60_1_6_1_0.25_0.75": 3,
        "60_1_15_1_0.25_0.75": 4,
    },  # center region, different No. of frames
    {
        "60_1_6_0_0.25_0.75": 1,
        "60_1_6_1_0.25_0.75": 2,
        "60_1_15_0_0.25_0.75": 3,
        "60_1_15_1_0.25_0.75": 4,
    },  # center region, consecutive vs uniformly
    {
        "60_1_6_0_0.25_0.75": 3,
        "60_3_6_0_0.25_0.75": 2,
        "60_5_6_0_0.25_0.75": 1,
        "60_5_15_0_0.25_0.75": 4,
    },  # center region, different window interval
    {
        "60_1_1_1_0.00_0.25": 1,
        "60_1_3_1_0.00_0.25": 2,
        "60_1_6_1_0.00_0.25": 3,
        "60_1_15_1_0.00_0.25": 4,
    },  # top region, different No. of frames
    {
        "60_1_6_0_0.00_0.25": 1,
        "60_1_6_1_0.00_0.25": 2,
        "60_1_15_0_0.00_0.25": 3,
        "60_1_15_1_0.00_0.25": 4,
    },  # top region, consecutive vs uniformly
    {
        "60_1_6_0_0.00_0.25": 4,
        "60_3_6_0_0.00_0.25": 3,
        "60_5_6_0_0.00_0.25": 1,
        "60_5_15_0_0.00_0.25": 2,
    },  # top region, different window interval
]

tearing_video_postfix_to_name = [
    {
        "60_1_0_0_0.00_0.00": "AllFrameNoTearing",
        "60_1_60_1_0.00_0.25": "AllFrameTopTearing",
        "60_1_60_1_0.25_0.75": "AllFrameMiddleTearing",
        "60_1_60_1_0.75_1.00": "AllFrameBottomTearing",
    },  # different positions, all frame tearing
    {
        "60_1_1_1_0.25_0.75": "1FrameMiddleTearing",
        "60_1_3_1_0.25_0.75": "3FrameMiddleTearing",
        "60_1_6_1_0.25_0.75": "6FrameMiddleTearing",
        "60_1_15_1_0.25_0.75": "15FrameMiddleTearing",
    },  # center region, different No. of frames
    {
        "60_1_6_0_0.25_0.75": "6FrameMiddleContinuousTearing",
        "60_1_6_1_0.25_0.75": "6FrameMiddleEvenTearing",
        "60_1_15_0_0.25_0.75": "15FrameMiddleContinuousTearing",
        "60_1_15_1_0.25_0.75": "15FrameMiddleEvenTearing",
    },  # center region, consecutive vs uniformly
    {
        "60_1_6_0_0.25_0.75": "6FrameMiddleTearing",
        "60_3_6_0_0.25_0.75": "6FrameEvery3sMiddleTearing",
        "60_5_6_0_0.25_0.75": "6FrameEvery5sMiddleTearing",
        "60_5_15_0_0.25_0.75": "15FrameEvery5sMiddleTearing",
    },  # center region, different window interval
    {
        "60_1_1_1_0.00_0.25": "1FrameTopTearing",
        "60_1_3_1_0.00_0.25": "3FrameTopTearing",
        "60_1_6_1_0.00_0.25": "6FrameTopTearing",
        "60_1_15_1_0.00_0.25": "15FrameTopTearing",
    },  # top region, different No. of frames
    {
        "60_1_6_0_0.00_0.25": "6FrameTopContinuousTearing",
        "60_1_6_1_0.00_0.25": "6FrameTopEvenTearing",
        "60_1_15_0_0.00_0.25": "15FrameTopContinuousTearing",
        "60_1_15_1_0.00_0.25": "15FrameTopEvenTearing",
    },  # top region, consecutive vs uniformly
    {
        "60_1_6_0_0.00_0.25": "6FrameTopTearing",
        "60_3_6_0_0.00_0.25": "6FrameEvery3sTopTearing",
        "60_5_6_0_0.00_0.25": "6FrameEvery5sTopTearing",
        "60_5_15_0_0.00_0.25": "15FrameEvery5sTopTearing",
    },  # top region, different window interval
]


single_anwser = collections.namedtuple(
    "single_anwser",
    "score_1 score_2 score_3 score_4 rank_1 rank_2 rank_3 rank_4 confidence",
)

user_info = collections.namedtuple("user_info", "qq phone age gender test_duration")


def read_video_identity_file(file_path, video_postfix_to_rank, video_postfix_to_name):
    lines = open(file_path).readlines()
    video_ranks = {}
    video_names = {}
    for i in range(len(lines) // 4 // 7):
        for k in range(7):
            ranks = []
            names = []
            video_name = None
            for j in range(4):
                line = lines[i * 28 + k * 4 + j]
                items = line.split(",")

                cur_video_name = items[0][:-6]
                if video_name == None:
                    video_name = cur_video_name
                elif cur_video_name != video_name:
                    raise

                vid = "_".join(items[1][:-5].split("_")[-6:])
                ranks.append(video_postfix_to_rank[k][vid])
                names.append(video_postfix_to_name[k][vid])
            video_ranks[video_name] = ranks
            video_names[video_name] = names

    return video_ranks, video_names


def read_user_record(record, games=["lostark"], video_no=2, question_no=7):
    info = user_info(
        qq=str(record[11]),
        phone=record[14][1:],
        age=int(record[13]),
        gender=int(record[12]),
        test_duration=int(record[3]),
    )

    results = {}
    st_idx = 16
    tot_no = 1 + video_no * question_no * 9
    for i in range(len(games)):
        experience = int(record[st_idx + i * tot_no])
        vid_cnt = 0

        game_result = {}
        for j in range(video_no):

            for k in range(question_no):
                vid_cnt += 1
                anwser = single_anwser(
                    score_1=int(
                        record[st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9]
                    ),
                    score_2=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 1
                        ]
                    ),
                    score_3=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 2
                        ]
                    ),
                    score_4=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 3
                        ]
                    ),
                    rank_1=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 4
                        ]
                    ),
                    rank_2=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 5
                        ]
                    ),
                    rank_3=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 6
                        ]
                    ),
                    rank_4=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 7
                        ]
                    ),
                    confidence=int(
                        record[
                            st_idx + i * tot_no + 1 + j * question_no * 9 + k * 9 + 8
                        ]
                    ),
                )

                question_id = "_".join([games[i], str(vid_cnt)])

                game_result[question_id] = anwser

        results[games[i]] = {"experience": experience, "result": game_result}

    return [info, results]


def read_all_user_records(file_path):
    df = pd.read_csv(file_path)
    records = []
    for i in range(0, len(df)):
        records.append(read_user_record(df.iloc[i], ["lostark"], 2, 7))

    return records


def clean_user_record(user_record, video_names, video_ranks):
    user_info, records = user_record

    anwsers = records["lostark"]["result"]
    cleaned_result = []
    for video_name in anwsers.keys():
        anwser_value = list(anwsers[video_name])
        reordered_name = []
        reordered_score = []
        reordered_rank = []
        cur_rank = np.argsort(anwser_value[4:8]).tolist()
        for i in cur_rank:
            reordered_name.append(video_names[video_name][i])
            reordered_rank.append(video_ranks[video_name][i])
            reordered_score.append(anwser_value[i])

        cleaned_result.append(
            [
                video_name,
                reordered_name,
                reordered_rank,
                reordered_score,
                anwser_value[8],
            ]
        )

    return user_info, cleaned_result


def cal_spearman_rank_coef(x, y):
    x = np.array(x)
    y = np.array(y)
    n = x.size

    return 1 - 6 * np.power((x - y), 2).sum() / n / (n**2 - 1)


def cal_user_accuracy(user_record, groundtruth_ranks):
    user_info, records = user_record
    acc_sum = 0
    conf_sum = 0
    cnt = 0

    anwsers = records["lostark"]["result"]
    for video_name in anwsers.keys():
        anwser_value = list(anwsers[video_name])
        acc = cal_spearman_rank_coef(groundtruth_ranks[video_name], anwser_value[4:8])

        acc_sum += acc
        conf_sum += anwser_value[-1]
        cnt += 1

    if cnt < 14:
        raise
    results = [user_info, acc_sum / cnt, conf_sum / cnt]
    return results


def cal_video_accuracy(user_records, video_ranks, video_names):
    results = []
    video_no = len(video_ranks)
    user_no = len(user_records)

    for i in range(video_no):
        cur_video_name = "lostark_%d" % (i + 1)
        cur_video_idx = np.argsort(video_ranks[cur_video_name])
        cur_video_scores = np.zeros(4)
        cur_video_rank = np.zeros(4)
        cur_video_conf = 0
        for j in range(user_no):
            user_info, anwsers = user_records[j]
            cur_video_rank += np.array(
                anwsers["lostark"]["result"][cur_video_name][4:8]
            )
            cur_video_scores += np.array(
                anwsers["lostark"]["result"][cur_video_name][:4]
            )
            cur_video_conf += anwsers["lostark"]["result"][cur_video_name][8]

        cur_video_rank /= user_no
        cur_video_scores /= user_no
        cur_video_conf /= user_no
        reordered_names = []
        for k in cur_video_idx.tolist():
            reordered_names.append(video_names[cur_video_name][k])

        results.append(
            [
                cur_video_name,
                reordered_names,
                cur_video_scores[cur_video_idx],
                cur_video_rank[cur_video_idx],
                cur_video_conf,
            ]
        )

    return results


if __name__ == "__main__":
    user_records = read_all_user_records(
        "test_data/tearing/14175903_202403261623323292.csv"
    )

    video_ranks, video_names = read_video_identity_file(
        "test_data/tearing/game_video_identity.csv",
        tearing_video_postfix_to_rank,
        tearing_video_postfix_to_name,
    )

    # print(user_records[0])
    # print(json.dumps(user_records[0], indent=4))
    # print(video_ranks['lostark_1'], video_names['lostark_1'])
    # print(cal_user_accuracy(user_records[0], video_ranks))
    # print(video_ranks)
    # print(clean_user_record(user_records[0], video_names, video_ranks))
    # for user_record in user_records:
    #     print(cal_user_accuracy(user_record, video_ranks))

    out_file_path = open("test_data/tearing/cleaned_user_records.csv", "w")
    out_file_path.write("QQ,Age,Gender,Test duration,Experience,Accuracy,Confidence\n")
    for user_record in user_records:
        acc_result = cal_user_accuracy(user_record, video_ranks)
        user_info, cleaned_anwsers = clean_user_record(
            user_record, video_names, video_ranks
        )
        result = ""
        result += ",".join(
            [
                str(item)
                for item in list(user_info)
                + [user_record[1]["lostark"]["experience"]]
                + acc_result[1:]
            ]
        )

        for anwser in cleaned_anwsers:
            cur_video_names = ",".join([anwser[0] + "_" + item for item in anwser[1]])
            cur_video_ranks = ",".join(
                [str(item) for item in anwser[2] + anwser[3] + [anwser[4]]]
            )

            result = result + "," + cur_video_names + "," + cur_video_ranks

        out_file_path.write(result + "\n")

    cleaned_video_results = cal_video_accuracy(user_records, video_ranks, video_names)
    out_file_path = open("test_data/tearing/cleaned_video_records.csv", "w")
    out_file_path.write(
        "video_name,tearing_patter_1,tearing_patter_2,tearing_patter_3,tearing_patter_4,score_1,score_2,score_3,score_4,rank_1,rank_2,rank_3,rank_4\n"
    )
    for video_record in cleaned_video_results:
        cur_video_names = ",".join(video_record[1])
        cur_video_scores = ",".join([str(item) for item in video_record[2]])
        cur_video_ranks = ",".join([str(item) for item in video_record[3]])
        result = (
            video_record[0]
            + ","
            + cur_video_names
            + ","
            + cur_video_scores
            + ","
            + cur_video_ranks
        )

        out_file_path.write(result + "\n")
