# date = 2021-11-18
# author = liuwei
import copy
import os
import json
import random
import csv

random.seed(106524)

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
    label_map = {"0": "1", "1": "2", "2": "3"}
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
            with open(train_file, "w", encoding="utf-8") as f:
                for text in prompt_fold_dict[token]:
                    f.write("%s\n"%(text))
            dev_file = os.path.join(fold_dir, "dev.json")
            token = "{}+{}+{}".format(prompt, fold_id, "valid")
            print("%s: %d" % (token, len(prompt_fold_dict[token])))
            with open(dev_file, "w", encoding="utf-8") as f:
                for text in prompt_fold_dict[token]:
                    f.write("%s\n" % (text))
            test_file = os.path.join(fold_dir, "test.json")
            token = "{}+{}+{}".format(prompt, fold_id, "test")
            print("%s: %d" % (token, len(prompt_fold_dict[token])))
            with open(test_file, "w", encoding="utf-8") as f:
                for text in prompt_fold_dict[token]:
                    f.write("%s\n" % (text))
        

def toefl_csv_reader_v1(file_name, out_dir):

    csv_reader = csv.reader(open(file_name))
    fold_id = int(file_name.split("/")[-1].split("_")[-1].split(".")[0])
    all_texts = []
    for idx in range(1, 9):
        all_texts.append([])
    score_map = {"0": "low", "1": "medium", "2": "high"}
    cur_line_id = 0
    for line in csv_reader:
        cur_line_id += 1
        if cur_line_id == 1:
            continue

        essay_id = line[0]
        prompt = line[1]
        text = line[4]
        score = line[3]

        all_texts[int(prompt)-1].append((essay_id, text, score))
    for idx in range(1, 9):
        cur_dir_name = os.path.join(out_dir, "toefl_p{}".format(idx))
        os.makedirs(cur_dir_name, exist_ok=True)
        cur_fold_dir = os.path.join(cur_dir_name, str(fold_id+1))
        os.makedirs(cur_fold_dir, exist_ok=True)

        out_name = file_name.split("/")[-1].split("_")[0]
        out_name = "{}.json".format(out_name)
        out_name = os.path.join(cur_fold_dir, out_name)

        with open(out_name, "w", encoding="utf-8") as f:
            for idy, item in enumerate(all_texts[idx-1]):
                sample = {}
                sample["id"] = idy
                sample["essay_id"] = item[0]
                sample["text"] = item[1]
                sample["score"] = score_map[item[2]]

                f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))

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
    """
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
    """
    # dataset = "Clinton"
    # gcdc_csv_reader(dataset, "train")

    ## preprocess toefl
    toefl_csv_reader()

