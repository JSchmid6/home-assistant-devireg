import argparse
import json
import os

import numpy as np
import pandas as pd


def preprocess_beijing(info_path):
    with open(f"{info_path}/beijing.json", "r") as f:
        info = json.load(f)

    data_path = info["raw_data_path"]

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info["data_path"], index=False)


def preprocess_news(info_path, raw_data_dir):
    with open(f"{info_path}/news.json", "r") as f:
        info = json.load(f)

    data_path = info["raw_data_path"]
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop("url", axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12, 18))]
    cat_columns2 = columns[list(range(30, 38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis=1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis=1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df["data_channel"] = cat_col1
    data_df["weekday"] = cat_col2

    data_save_path = f"{raw_data_dir}/news/news.csv"
    data_df.to_csv(f"{data_save_path}", index=False)

    columns = np.array(data_df.columns.tolist())

    info["num_col_idx"] = list(range(45))
    info["cat_col_idx"] = [46, 47]
    info["target_col_idx"] = [45]
    info["data_path"] = data_save_path

    name = "news"
    with open(f"{info_path}/{name}.json", "w") as file:
        json.dump(info, file, indent=4)


def get_column_name_mapping(
    data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
):
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 2024

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1

    return train_df, test_df, seed


def process_data(name, info_path, data_dir):
    raw_data_dir = os.path.join(data_dir, "raw_data")
    processed_data_dir = os.path.join(data_dir, "processed_data")

    if name == "news":
        preprocess_news(info_path, raw_data_dir)
    elif name == "beijing":
        preprocess_beijing(info_path)

    with open(f"{info_path}/{name}.json", "r") as f:
        info = json.load(f)

    data_path = info["data_path"] # SMK: Here it reads the input file based on the path given in the json file
    if info["file_type"] == "csv":
        data_df = pd.read_csv(data_path, header=info["header"])

    elif info["file_type"] == "xls":
        data_df = pd.read_excel(data_path, sheet_name="Data", header=1)
        data_df = data_df.drop("ID", axis=1)

    num_data = data_df.shape[0]

    column_names = (
        info["column_names"] if info["column_names"] else data_df.columns.tolist()
    )

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    if info["test_path"]:
        # if testing data is given
        test_path = info["test_path"]
        test_df = pd.read_csv(test_path)
        train_df = data_df

        # with open(test_path, "r") as f:
        #     lines = f.readlines()[1:]
        #     test_save_path = f"{raw_data_dir}/{name}/test.data"
        #     if not os.path.exists(test_save_path):
        #         with open(test_save_path, "a") as f1:
        #             for line in lines:
        #                 save_line = line.strip("\n").strip(".")
        #                 f1.write(f"{save_line}\n")
    else:
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set) #SMK split is 90% train, 10% test
        num_train = int(num_data * 0.99)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(
            data_df, cat_columns, num_train, num_test
        )

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())
        col_info["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == "?", col] = "nan"

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy().astype(np.int64)
    y_train = train_df[target_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy().astype(np.int32)
    y_test = test_df[target_columns].to_numpy()

    if not os.path.exists(f"{processed_data_dir}/{name}"):
        os.makedirs(f"{processed_data_dir}/{name}")

    np.save(f"{processed_data_dir}/{name}/X_num_train.npy", X_num_train)
    np.save(f"{processed_data_dir}/{name}/X_cat_train.npy", X_cat_train)
    np.save(f"{processed_data_dir}/{name}/y_train.npy", y_train)

    np.save(f"{processed_data_dir}/{name}/X_num_test.npy", X_num_test)
    np.save(f"{processed_data_dir}/{name}/X_cat_test.npy", X_cat_test)
    np.save(f"{processed_data_dir}/{name}/y_test.npy", y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f"{processed_data_dir}/{name}/train.csv", index=False)
    test_df.to_csv(f"{processed_data_dir}/{name}/test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{processed_data_dir}/{name}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and Saving {name} Successfully!")

    print("Dataset Name:", name)
    print("Total Size:", info["train_num"] + info["test_num"])
    print("Train Size:", info["train_num"])
    print("Test Size:", info["test_num"])
    if info["task_type"] == "regression":
        num = len(info["num_col_idx"] + info["target_col_idx"])
        cat = len(info["cat_col_idx"])
    else:
        cat = len(info["cat_col_idx"] + info["target_col_idx"])
        num = len(info["num_col_idx"])
    print("Number of Numerical Columns:", num)
    print("Number of Categorical Columns:", cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset")

    # General configs
    parser.add_argument("--dataname", type=str, default=None, help="Name of dataset.")
    args = parser.parse_args()

    INFO_PATH = "data/Info"
    DATA_DIR = "/projects/aieng/diffusion_bootcamp/data/tabular"

    if args.dataname:
        process_data(args.dataname, INFO_PATH, DATA_DIR)
    else:
        for name in ["adult", "default", "shoppers", "magic", "beijing", "news"]:
            process_data(name, INFO_PATH, DATA_DIR)
