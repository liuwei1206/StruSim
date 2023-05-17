# date = 2021-11-18
# author = liuwei
import copy
import os
import json
import random

random.seed(106524)

def k_fold_for_gcdc(data_dir, dataset):
    """
    k fold split data for dataset
    Args:
        data_dir: data/dataset
        dataset: clinton, enron, yelp, yahoo
    """
    ori_data_dir = os.path.join(data_dir, dataset)
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

        group_data_dir = os.path.join("k_fold_gcdc", dataset)
        group_data_dir = os.path.join(group_data_dir, str(idx))
        os.makedirs(group_data_dir, exist_ok=True)
        tmp_train_file = os.path.join(group_data_dir, "train.json")
        tmp_dev_train_file = os.path.join(group_data_dir, "dev.json")

        count = 0
        with open(tmp_train_file, 'w', encoding="utf-8") as f:
            for line in train_samples:
                sample = json.loads(line)
                sample["id"] = count
                f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))
                count += 1

        with open(tmp_dev_train_file, "w", encoding="utf-8") as f:
            for line in dev_samples:
                sample = json.loads(line)
                sample["id"] = count
                f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))
                count += 1

def k_prompt_for_toefl(data_dir):
    """
    split data for toelf dataset with different prompt, each prompt with 5-fold crossvalidation
    Args:
        data_dir:
        prompt_id:
    """
    files = os.listdir(data_dir)
    files = sorted(files)
    files = [os.path.join(data_dir, f) for f in files if ".json" in f]
    prompt_num = len(files)

    # read data from file
    all_prompts_categories = [[[] for _ in range(3)] for _ in range(prompt_num)]
    for prompt_id, file_name in enumerate(files):
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    text = sample['text']
                    label = sample['score']
                    prompt = sample['prompt']

                    if label.lower() == 'low':
                        all_prompts_categories[prompt_id][0].append((prompt, text, label))
                    elif label.lower() == 'medium':
                        all_prompts_categories[prompt_id][1].append((prompt, text, label))
                    elif label.lower() == 'high':
                        all_prompts_categories[prompt_id][2].append((prompt, text, label))

    # split
    for prompt_id in range(prompt_num):
        tmp_all_prompts_categories = None
        tmp_all_prompts_categories = copy.deepcopy(all_prompts_categories)

        train_dev_high = []
        train_dev_medium = []
        train_dev_low = []

        for idx in range(prompt_num):
            if idx == prompt_id:
                continue
            else:
                train_dev_low.extend(tmp_all_prompts_categories[idx][0])
                train_dev_medium.extend(tmp_all_prompts_categories[idx][1])
                train_dev_high.extend(tmp_all_prompts_categories[idx][2])

        random.shuffle(train_dev_low)
        random.shuffle(train_dev_medium)
        random.shuffle(train_dev_high)
        print("low num: ", len(train_dev_low))
        print("medium num: ", len(train_dev_medium))
        print("high num: ", len(train_dev_high))
        # 5-fold cross validation
        five_group_sample = [[] for _ in range(5)]
        pivot = 0
        for item in train_dev_low:
            five_group_sample[pivot].append(item)
            pivot += 1
            pivot = pivot % 5

        for item in train_dev_medium:
            five_group_sample[pivot].append(item)
            pivot += 1
            pivot = pivot % 5

        for item in train_dev_high:
            five_group_sample[pivot].append(item)
            pivot += 1
            pivot = pivot % 5


        # write one group into dev.json and other four groups into train.json
        dir_name = os.path.join("k_fold_toefl", "toefl_p{}".format(prompt_id+1))
        os.makedirs(dir_name, exist_ok=True)

        for idx in range(5):
            cur_dir_name = os.path.join(dir_name, "{}".format(idx+1))
            os.makedirs(cur_dir_name, exist_ok=True)
            cur_train_file = os.path.join(cur_dir_name, "train.json")
            cur_dev_file = os.path.join(cur_dir_name, "dev.json")
            train_samples = []
            dev_samples = []
            for idy in range(5):
                if idx == idy:
                    dev_samples.extend(five_group_sample[idy])
                else:
                    train_samples.extend(five_group_sample[idy])

            # for train
            count = 0
            with open(cur_train_file, "w", encoding="utf-8") as f:
                for sample in train_samples:
                    now_sample = {}
                    now_sample['id'] = count
                    now_sample['prompt'] = sample[0]
                    now_sample['text'] = sample[1]
                    now_sample['score'] = sample[2]
                    count += 1

                    f.write("%s\n"%(json.dumps(now_sample, ensure_ascii=False)))

            # for dev
            count = 0
            with open(cur_dev_file, "w", encoding="utf-8") as f:
                for sample in dev_samples:
                    now_sample = {}
                    now_sample['id'] = count
                    now_sample['prompt'] = sample[0]
                    now_sample['text'] = sample[1]
                    now_sample['score'] = sample[2]
                    count += 1

                    f.write("%s\n" % (json.dumps(now_sample, ensure_ascii=False)))


if __name__ == "__main__":
    ## preprocess gcdc
    data_dir = "data/dataset"
    dataset_list = ["gcdc_yahoo", "gcdc_clinton", "gcdc_enron", "gcdc_yelp"]
    for dataset in dataset_list:
        print(" Preprocessing %s\n"%(dataset))
        # k_fold_for_gcdc(data_dir, dataset)

    ## preprocess toefl
    data_dir = "data/dataset/toefl"
    # k_prompt_for_toefl(data_dir)

