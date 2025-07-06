"""Module providing load functionality of pysdg."""

import sys
import copy
import os
import json
from typing import Tuple
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import torch
import gc


def _update_safe_info(new_col_names, raw_info):
    safe_info = copy.deepcopy(raw_info)
    safe_info["org_names_ord"] = new_col_names
    safe_info["cat_names"] = []
    safe_info["cnt_names"] = []
    safe_info["dscrt_names"] = []
    # safe_info['const_names']=[]
    safe_info["datetime_names"] = []
    safe_info["text_names"] = []
    safe_info["quasi_names"] = []
    for new_col in new_col_names:
        if new_col.split("%%%")[0] in raw_info["cat_names"]:
            safe_info["cat_names"].append(new_col)
        elif new_col.split("%%%")[0] in raw_info["cnt_names"]:
            safe_info["cnt_names"].append(new_col)
        elif new_col.split("%%%")[0] in raw_info["dscrt_names"]:
            safe_info["dscrt_names"].append(new_col)
        # elif new_col.split('%%%')[0] in raw_info['const_names']:
        #     safe_info['const_names'].append(new_col)
        elif new_col.split("%%%")[0] in raw_info["datetime_names"]:
            safe_info["datetime_names"].append(new_col)
        elif new_col.split("%%%")[0] in raw_info["text_names"]:
            safe_info["text_names"].append(new_col)
    for new_col in new_col_names:
        if new_col.split("%%%")[0] in raw_info["quasi_names"]:
            safe_info["quasi_names"].append(new_col)
    return safe_info


def _get_delta_days(
    input_str: str, var_name: str, anchor_date: pd.Timestamp, suppress_errors: bool
):
    try:
        input_date = pd.to_datetime(input_str, errors="raise")
        delta = (input_date.date() - anchor_date.date()).days
        return delta
    except ValueError as exc:
        if "%%MISS" in input_str:
            return input_str
        else:
            if suppress_errors:
                return "%%ERR"
            else:
                raise TypeError(
                    f"Only dates are allowed for variable: {var_name}"
                ) from exc


def _get_anchor(df: pd.DataFrame) -> str:
    # Flatten the DataFrame (includes only datetim variables) to a single series
    all_entries = df.values.flatten()

    # Convert entries to datetime, ignoring errors
    datetimes = pd.to_datetime(all_entries, errors="coerce")

    # Filter out NaT values (invalid datetimes)
    valid_datetimes = datetimes[~pd.isnull(datetimes)]

    # Check if there are any valid datetimes
    if len(valid_datetimes) > 0:
        # Return the smallest datetime
        return valid_datetimes.min()
    else:
        # Return None if no valid datetimes are found
        return None


def _label_erratic_values(
    var_name: str, safe_data: pd.DataFrame, suppress_errors: bool
):
    def convert_value(value, var_name, suppress_errors):
        try:
            return float(value)
        except ValueError as exc:
            if suppress_errors:
                return "%%ERR"
            else:
                raise TypeError(
                    f"Only numeric values are allowed for the variable: {var_name}"
                ) from exc

    safe_data[var_name] = safe_data[var_name].apply(
        lambda x: (
            convert_value(x, var_name, suppress_errors) if "%%MISS" not in x else x
        )
    )
    return safe_data


def _nullify_erratic_values(
    naive_info: dict, dataset: pd.DataFrame, suppress_errors: bool
):
    for var_name in dataset.columns:
        if var_name in naive_info["datetime_names"]:
            dataset[var_name] = pd.to_datetime(
                dataset[var_name], errors="coerce" if suppress_errors else "raise"
            )
        elif (
            var_name in naive_info["cnt_names"] or var_name in naive_info["dscrt_names"]
        ):
            try:
                dataset[var_name] = pd.to_numeric(
                    dataset[var_name], errors="coerce" if suppress_errors else "raise"
                )
                dataset[var_name] = pd.to_numeric(dataset[var_name], downcast="float")
            except ValueError as exc:
                if not suppress_errors:
                    raise TypeError(
                        f"Only numeric values are allowed for variable: {var_name}"
                    ) from exc
    return dataset


def _encode_missings(
    raw_data: pd.DataFrame, raw_info: dict, suppress_errors: bool
) -> Tuple[pd.DataFrame, dict]:
    """The loader allows to safely load the input dataframe or a subset of it
    into generator-friendly dataset by proper handling of missing values.

    Args:
        raw_data: the input raw tabular dataset
        raw_info: the information associated with input dataset including variable names, types and missing value codes.
        suppress_errors: A boolean that, if set to True, activates the detection of erratic values in columns and treating them as missing values.

    Returns:
        Tuple[pd.DataFrame, dict]: a dataset that can be safely used in training generative models.
    """

    raw_info = _get_var_names(raw_data, raw_info)
    org_col_names = list(raw_data.columns)

    safe_data = copy.deepcopy(raw_data).astype("str")
    new_col_names = [
        f"{col_name}%%%{i}" for i, col_name in enumerate(list(safe_data.columns))
    ]

    safe_data.columns = new_col_names
    safe_info = _update_safe_info(new_col_names, raw_info)
    safe_info["num_names_w_miss"] = []
    safe_info["num_names_w_err"] = []

    # 1) Replacing all missing values by pysdg names
    new_miss_vals = []
    for i, _ in enumerate(raw_info["miss_vals"]):
        new_miss_vals.append(f"%%MISS-{i}%%")
    miss_map = dict(zip(raw_info["miss_vals"], new_miss_vals))
    safe_data = safe_data.replace(miss_map)
    safe_info["safe_miss_vals"] = new_miss_vals

    # 2) Deal with categorical variables
    for var_name in safe_info["cat_names"]:
        safe_data[var_name] = safe_data[var_name].astype(
            "category"
        )  # Enforce type  for categorical vars

    # 3) Deal with continuous (float)
    for var_name in safe_info["cnt_names"]:
        safe_data = _label_erratic_values(var_name, safe_data, suppress_errors)
        safe_data, safe_info = _impute(var_name, safe_data, safe_info)
        safe_data[var_name] = safe_data[var_name].astype("float")

    # 4) Deal with discrete integers (can not take values other than indicated)
    for var_name in safe_info["dscrt_names"]:
        safe_data = _label_erratic_values(var_name, safe_data, suppress_errors)
        safe_data, safe_info = _impute(var_name, safe_data, safe_info)
        safe_data[var_name] = safe_data[var_name].astype(
            "float"
        )  # using float will avoid errors in some generators

    # 5) Process datetime variables
    if len(safe_info["datetime_names"]) > 0:
        anchor = _get_anchor(safe_data[safe_info["datetime_names"]])
        safe_info["anchor_date"] = anchor
        for var_name in safe_info["datetime_names"]:
            safe_data[var_name] = safe_data[var_name].apply(
                lambda x, var=var_name: _get_delta_days(
                    x, var, anchor, suppress_errors=suppress_errors
                )
            )
            safe_data[var_name] = safe_data[var_name].astype("str")
            safe_data, safe_info = _impute(var_name, safe_data, safe_info)
            safe_data[var_name] = safe_data[var_name].astype(
                "float"
            )  # using float will avoid errors in some generators

    safe_info["dropped_names"] = []
    # 6) Drop id column
    for var_name in safe_info["id_name"]:
        id_var_idx = org_col_names.index(var_name)
        new_var_name = safe_data.columns[id_var_idx]
        safe_data = safe_data.drop(new_var_name, axis=1)
        safe_info["dropped_names"].append(new_var_name)

    # 7) Drop constant variables
    for col_name, col_data in safe_data.items():
        if len(col_data.unique()) == 1:
            safe_data = safe_data.drop(col_name, axis=1)
            safe_info["dropped_names"].append(col_name)

    # 8) Drop all null variables
    safe_info["null_col_names"] = []  # add the null column names to info
    for col_name, col_data in safe_data.items():
        if col_data.isnull().values.all():
            safe_data = safe_data.drop(col_name, axis=1)
            safe_info["dropped_names"].append(var_name)

    return safe_data, safe_info


def _get_var_names(data: pd.DataFrame, info: dict) -> dict:
    if set(
        [
            "id_idx",
            "cat_idxs",
            "cnt_idxs",
            "dscrt_idxs",
            "datetime_idxs",
            "text_idxs",
            "quasi_idxs",
        ]
    ).issubset(set(info.keys())):
        info["id_name"] = list(data.columns[info["id_idx"]])
        del info["id_idx"]
        info["cat_names"] = list(data.columns[info["cat_idxs"]])
        del info["cat_idxs"]
        info["cnt_names"] = list(data.columns[info["cnt_idxs"]])
        del info["cnt_idxs"]
        info["dscrt_names"] = list(data.columns[info["dscrt_idxs"]])
        del info["dscrt_idxs"]
        info["datetime_names"] = list(data.columns[info["datetime_idxs"]])
        del info["datetime_idxs"]
        info["text_names"] = list(data.columns[info["text_idxs"]])
        del info["text_idxs"]
        info["quasi_names"] = list(data.columns[info["quasi_idxs"]])
        del info["quasi_idxs"]
    return info


def _impute(
    var_name: str, safe_data: pd.DataFrame, safe_info: dict
) -> Tuple[pd.DataFrame, dict]:
    na_idxs = list(
        safe_data[safe_data[var_name].astype(str).str.contains("%%MISS")].index
    )
    err_idxs = list(
        safe_data[safe_data[var_name].astype(str).str.contains("%%ERR")].index
    )
    if (len(na_idxs) == 0) and (len(err_idxs) == 0):
        return safe_data, safe_info
    if len(na_idxs) > 0:
        safe_info["num_names_w_miss"].append(var_name)
        safe_data.loc[na_idxs, var_name] = (
            np.nan
        )  # Can not use pd.NA since SimpleImputer will throw and error
    if len(err_idxs) > 0:
        safe_info["num_names_w_err"].append(var_name)
        safe_data.loc[err_idxs, var_name] = (
            np.nan
        )  # erratic values will be dealt with as missing values for numeric (i.e. cnt, discete and datetime) variables.
    if var_name in safe_info["cnt_names"]:
        imp = SimpleImputer(missing_values=pd.NA, strategy="mean")
    elif var_name in safe_info["dscrt_names"]:
        imp = SimpleImputer(missing_values=pd.NA, strategy="most_frequent")
    elif var_name in safe_info["datetime_names"]:
        imp = SimpleImputer(missing_values=pd.NA, strategy="mean")
    safe_data[f"{var_name}_missing"] = np.zeros(
        len(safe_data)
    )  # create dummy column to keep track of NA in numeric variables
    safe_data.loc[na_idxs, f"{var_name}_missing"] = 1
    safe_data.loc[err_idxs, f"{var_name}_missing"] = 1
    safe_data[f"{var_name}_missing"] = safe_data[f"{var_name}_missing"].astype("bool")
    imputed_array = imp.fit_transform(
        safe_data[var_name].values.reshape(-1, 1)
    )  # impute the numeric variable here
    safe_data.loc[:, var_name] = (
        imputed_array if imputed_array.shape[1] != 0 else safe_data.loc[:, var_name]
    )  # update the numeric var with imputed nan values
    return safe_data, safe_info


def _convert_na_str(col: pd.Series, na_values: list) -> pd.Series:
    if len(col.unique()) == 1:
        na_val = col.unique()[0]
        if na_val in na_values:
            col = col.replace(na_val, np.NaN)
            return col
        else:
            return col
    else:
        return col


def _decode_missings(
    raw_data: pd.DataFrame, safe_data: pd.DataFrame, safe_info: dict, impute: bool
) -> Tuple[pd.DataFrame, dict]:
    restored_data = copy.deepcopy(safe_data)
    restored_info = copy.deepcopy(safe_info)
    # restored_data_var_names=list(restored_data.columns)

    # 1) Restore all dropped variables
    for var_name in restored_info["dropped_names"]:
        restored_data[var_name] = raw_data[var_name.split("%%%")[0]]

    # 2) Restore all missing values
    inv_miss_map = dict(
        zip(restored_info["safe_miss_vals"], restored_info["miss_vals"])
    )
    restored_data = restored_data.replace(
        inv_miss_map
    )  # map all encoded missing values to original missing
    restored_data_w_org_miss = copy.deepcopy(restored_data)

    # 3) restore continuous variables
    for var_name in restored_info["cnt_names"]:
        if var_name in restored_info["num_names_w_miss"]:
            #print(var_name)
            missing_indxs = np.where(restored_data[f"{var_name}_missing"].astype(bool))[0].tolist()
            restored_data.loc[missing_indxs, var_name] = (
                np.NaN
                if not impute
                else restored_data_w_org_miss.loc[missing_indxs, var_name]
            )
            restored_data = restored_data.drop(f"{var_name}_missing", axis=1)
        restored_data[var_name] = restored_data[var_name].astype(
            "float"
        )  # if not as_strings else restored_data[var_name].astype('str')

    # 4) restore discrete variables
    for var_name in restored_info["dscrt_names"]:
        restored_data[var_name] = _convert_na_str(
            restored_data[var_name], restored_info["miss_vals"]
        )
        restored_data[var_name] = restored_data[var_name].astype("float").round(0)
        if var_name in restored_info["num_names_w_miss"]:
            #print(var_name)
            missing_indxs = np.where(restored_data[f"{var_name}_missing"].astype(bool))[0].tolist()
            #missing_indxs = list(restored_data[restored_data[f"{var_name}_missing"].astype(bool)].index)  # get indexes with True values reflecting missing
            restored_data.loc[missing_indxs, var_name] = (
                np.NaN
                if not impute
                else restored_data_w_org_miss.loc[missing_indxs, var_name]
            )
            restored_data = restored_data.drop(f"{var_name}_missing", axis=1)
        restored_data[var_name] = restored_data[var_name].astype(
            "float"
        )  # if not as_strings else restored_data[var_name].astype('str')

    # 5) restore datetime variables
    anchor = (
        pd.to_datetime(restored_info["anchor_date"])
        if "anchor_date" in restored_info.keys()
        else None
    )  # first restore using anchor date then restore the missing values
    for var_name in restored_info["datetime_names"]:
        if var_name in restored_info["num_names_w_miss"]:
            #print(var_name)
            missing_indxs = np.where(restored_data[f"{var_name}_missing"].astype(bool))[0].tolist()
            #missing_indxs = list(restored_data[restored_data[f"{var_name}_missing"].astype(bool)].index)  # get indexes with True values reflecting missing
            restored_data.loc[missing_indxs, var_name] = np.NaN
            restored_data = restored_data.drop(f"{var_name}_missing", axis=1)
        restored_data[var_name] = pd.to_timedelta(restored_data[var_name], "D")
        restored_data[var_name] = (restored_data[var_name] + anchor).dt.date

    # 6) assert that raw data and restored data have same shape
    restored_data = restored_data[
        safe_info["org_names_ord"]
    ]  # ensure that the variables in the restored data are ordered in the same way of of the raw data
    restore_col_names = [
        f'{col_name.split("%%%")[0]}' for col_name in restored_data.columns
    ]
    restored_data.columns = restore_col_names
    assert restored_data.shape == raw_data.shape
    return restored_data, restored_info


def naive_write(df: pd.DataFrame, dir_uri: str, fname: str):
    df = df.astype("str")
    if not os.path.exists(dir_uri):
        os.makedirs(dir_uri)
    df.to_csv(os.path.join(dir_uri, fname), index=False)
    print(f"file saved to {os.path.join(dir_uri,fname)}")


def do_sweep_replica(replica_client: object, input_replica_ids: list) -> list:
    """Enforces deletions of Replica created files and folders that were
    generated in the training and generation processes.

    Arg:
        input_replica_ids: A list of the job ids and simulator ids.

    Return:
        A list with the result of deleting the folders associated with each of the input Replica job id.
    """
    res = []
    for job_id in input_replica_ids:
        if "simulator" not in job_id:
            res.append(replica_client.delete_job(job_id))
        else:
            res.append(replica_client.delete_simulator(job_id))
    return res


def _get_caller_directory():
    if "ipykernel" in sys.modules:
        # If running in a Jupyter notebook
        return os.getcwd()
    else:
        # If running in a regular Python file
        frame = sys._getframe(1)
        caller_file = frame.f_globals.get("__file__")
        if caller_file:
            return os.path.dirname(os.path.abspath(caller_file))
        else:
            raise FileNotFoundError(
                "No __file__ found and not in a notebook environment."
            )


def _set_working_directory():
    caller_directory = _get_caller_directory()
    os.chdir(caller_directory)


def _load_env_file():
    _set_working_directory()
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found in the directory: {os.getcwd()}")
    load_dotenv(env_path, verbose=True, override=True)


def inspect_data(path_to_csv: str, path_to_json: str):
    """Check indexes, variable names, data types and unique values for each
    variable in the input dataset and it metadata. A text file showing the
    results is generated carrying the name of teh dataset as extracted from the
    metadata file.

    Args:
        path_to_csv: The full path to the input dataset csv file.
        path_to_json: The full path to the metadata json file.
    """
    data = pd.read_csv(path_to_csv, dtype=str)
    with open(path_to_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    with open(f'{info["ds_name"]}.txt', "w", encoding="utf-8") as file:
        file.write(
            f"****************** DATASET: {info['ds_name']}**********************\n\n"
        )

        file.write("****************** CATEGORICAL VARIABLES **********************\n")
        for idx in info["cat_idxs"]:
            file.write(
                f"Index {idx}/{data.columns[idx]}/cat: {data.iloc[:,idx].unique()}\n\n"
            )

        file.write("****************** CONTINUOUS VARIABLES **********************\n")
        for idx in info["cnt_idxs"]:
            file.write(
                f"Index {idx}/{data.columns[idx]}/cnt : {data.iloc[:,idx].unique()}\n\n"
            )

        file.write("****************** DISCRETE VARIABLES **********************\n")
        for idx in info["dscrt_idxs"]:
            file.write(
                f"Index {idx}/{data.columns[idx]}/dscrt : {data.iloc[:,idx].unique()}\n\n"
            )

        file.write("****************** DATETIME VARIABLES **********************\n")
        for idx in info["datetime_idxs"]:
            file.write(
                f"Index {idx}/{data.columns[idx]}/datetime : {data.iloc[:,idx].unique()}\n\n"
            )


def clear_vars():
    """
    Clears non-essential user-defined variables from the global namespace,
    keeping important system libraries and functions intact.
    """
    # List of names you do not want to delete
    protected_vars = [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
        "gc",
        "torch",
        "sys",
        "os",
        "np",
        "pd",
    ]

    # Get a list of all global variables
    global_vars = (
        globals().copy()
    )  # Make a copy to avoid modifying the dictionary while iterating

    # Remove all variables except those in protected_vars
    for var_name in global_vars:
        if var_name not in protected_vars:
            del globals()[var_name]
            print(f"Deleted variable: {var_name}")

    # Force garbage collection
    gc.collect()

    # Clear any unused GPU memory if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared.")

################################### Dealing with json


def save_dict_to_json(data: dict, file_path: str):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): Full path including filename where JSON will be saved.

    Raises:
        ValueError: If the provided data is not a dictionary.
        IOError: If the file cannot be written.
    """
    if not isinstance(data, dict):
        raise ValueError("Provided data must be a dictionary.")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Dictionary successfully saved to: {file_path}")
    except IOError as e:
        print(f"Failed to write file: {e}")



def load_json_to_dict(file_path: str) -> dict:
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): Full path to the JSON file.

    Returns:
        dict: Dictionary containing the JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
