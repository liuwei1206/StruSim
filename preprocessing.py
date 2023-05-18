# date = 2021-11-18
# author = liuwei
import copy
import os
import json
import random
import csv
import stanza

random.seed(106524)

def write_array_into_file(file_name, all_texts):
    with open(file_name, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write("%s\n"%(text))

def gcdc_csv_reader(dataset, mode):
    """
    convert raw csv file into json

    Args:
        dataset: Clinton or Enron or Yahoo or Yelp
        mode: train or test
    """
    file_name = os.path.join("data/dataset/raw/gcdc", "{}_{}.csv".format(dataset, mode))
    out_dir = "data/dataset/gcdc_{}".format(dataset.lower())
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    csv_reader = csv.reader(open(file_name))
    cur_line_id = 0
    all_texts = []
    for line in csv_reader:
        cur_line_id += 1
        if cur_line_id == 1:
            text_idx = line.index("text")
            label_idx = line.index("labelA")
            continue
        sample = {}
        sample["id"] = cur_line_id - 2
        sample["score"] = line[label_idx]
        sample["text"] = line[text_idx]

        all_texts.append(json.dumps(sample, ensure_ascii=False))

    with open(out_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write("%s\n"%(text))

def toefl_csv_reader():
    label_map = {"0": "low", "1": "medium", "2": "high"}
    file_list = os.listdir("data/dataset/raw/toefl")
    prompt_fold_dict = {}
    for file_name in file_list:
        items = file_name.split(".")[0].split("_")
        mode = items[0]
        fold_id = int(items[2]) + 1
        csv_reader = csv.reader(open(os.path.join("data/dataset/raw/toefl", file_name)))
        cur_line = 0
        for line in csv_reader:
            cur_line += 1
            if cur_line == 1:
                essay_idx = line.index("essay_id")
                prompt_idx = line.index("prompt")
                label_idx = line.index("essay_score")
                text_idx = line.index("essay")
                continue
            prompt = int(line[prompt_idx])
            sample = {}
            sample["idx"] = line[essay_idx]
            sample["score"] = label_map[line[label_idx]]
            sample["text"] = line[text_idx]
            sample_text = json.dumps(sample, ensure_ascii=False)
            token = "{}+{}+{}".format(prompt, fold_id, mode)
            if token in prompt_fold_dict:
                prompt_fold_dict[token].append(sample_text)
            else:
                prompt_fold_dict[token] = [sample_text]

    for prompt in range(1, 9):
        data_dir = "data/dataset/toefl_p{}".format(prompt)
        os.makedirs(data_dir, exist_ok=True)
        for fold_id in range(1, 6):
            fold_dir = os.path.join(data_dir, str(fold_id))
            os.makedirs(fold_dir, exist_ok=True)

            train_file = os.path.join(fold_dir, "train.json")
            token = "{}+{}+{}".format(prompt, fold_id, "train")
            print("%s: %d" % (token, len(prompt_fold_dict[token])))
            write_array_into_file(train_file, prompt_fold_dict[token])

            dev_file = os.path.join(fold_dir, "dev.json")
            token = "{}+{}+{}".format(prompt, fold_id, "valid")
            print("%s: %d" % (token, len(prompt_fold_dict[token])))
            write_array_into_file(dev_file, prompt_fold_dict[token])

            test_file = os.path.join(fold_dir, "test.json")
            token = "{}+{}+{}".format(prompt, fold_id, "test")
            print("%s: %d" % (token, len(prompt_fold_dict[token])))
            write_array_into_file(test_file, prompt_fold_dict[token])

def k_fold_for_gcdc(dataset):
    """
    k fold split data for dataset
    Args:
        data_dir: data/dataset
        dataset: clinton, enron, yelp, yahoo
    """
    ori_data_dir = os.path.join("data/dataset", "gcdc_{}".format(dataset.lower()))
    train_file = os.path.join(ori_data_dir, "train.json")
    lows = []
    mediums = []
    highs = []
    with open(train_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                score = sample['score']

                if score == "1":
                    lows.append(line)
                elif score == "2":
                    mediums.append(line)
                elif score == "3":
                    highs.append(line)

    print("low num: %d, medium num: %d, high num: %d\n"%(len(lows), len(mediums), len(highs)))
    # we use 10 fold
    ten_group_samples = [[] for _ in range(10)]
    pivot = 0
    for sample in lows:
        ten_group_samples[pivot].append(sample)
        pivot += 1
        pivot = pivot % 10

    for sample in mediums:
        ten_group_samples[pivot].append(sample)
        pivot += 1
        pivot = pivot % 10

    for sample in highs:
        ten_group_samples[pivot].append(sample)
        pivot += 1
        pivot = pivot % 10

    _ = [random.shuffle(samples) for samples in ten_group_samples]

    # write ten files
    for idx in range(1, 11):
        train_samples = []
        dev_samples = []

        for idy in range(10):
            if idy + 1 == idx:
                dev_samples.extend(ten_group_samples[idy])
            else:
                train_samples.extend(ten_group_samples[idy])

        group_data_dir = os.path.join(ori_data_dir, str(idx))
        os.makedirs(group_data_dir, exist_ok=True)
        tmp_train_file = os.path.join(group_data_dir, "train.json")
        tmp_dev_train_file = os.path.join(group_data_dir, "dev.json")

        with open(tmp_train_file, 'w', encoding="utf-8") as f:
            for line in train_samples:
                sample = json.loads(line)
                f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))

        with open(tmp_dev_train_file, "w", encoding="utf-8") as f:
            for line in dev_samples:
                sample = json.loads(line)
                f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))


if __name__ == "__main__":
    ## 1.preprocess gcdc
    # """
    dataset_list = ["Clinton", "Enron", "Yahoo", "Yelp"]
    mode_list = ["train", "test"]
    for dataset in dataset_list:
        print("Processing %s ...."%(dataset))
        for mode in mode_list:
            gcdc_csv_reader(dataset, mode)
        # k_fold for train
        k_fold_for_gcdc(dataset)
        # copy test
        for idx in range(1, 11):
            command = "cp data/dataset/gcdc_{}/test.json data/dataset/gcdc_{}/{}/".format(
                dataset.lower(), dataset.lower(), str(idx)
            )
            os.system(command)
        # delete
        command = "rm -f data/dataset/gcdc_{}/train.json".format(dataset.lower())
        os.system(command)
        command = "rm -f data/dataset/gcdc_{}/test.json".format(dataset.lower())
        os.system(command)
    # """

    ## 2.preprocess toefl
    toefl_csv_reader()

    ## 3. download stanza
    stanza_dir = "data/stanza_resources"
    os.makedirs(stanza_dir, exist_ok=True)
    # stanza.download("en", model_dir=stanza_dir)
    
