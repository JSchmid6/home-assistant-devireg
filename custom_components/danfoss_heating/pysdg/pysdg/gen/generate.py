"""
This module provides a comprehensive implementation of a synthetic data generation framework. It includes a `Generator` class that supports multiple synthetic data generation techniques, including Replica, SynthCity, SDV, Yandex TabDDPM, and Amazon TabSyn. The module is designed to handle various data types and formats, ensuring compatibility and flexibility for different use cases.
"""

import os
import json
import warnings
import zipfile
import tempfile
import stat
import shutil
import uuid
import time
import pickle
from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
)
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from pysdg.config import configure_logging
from pysdg.load.load import (
    _decode_missings,
    _encode_missings,
    _get_var_names,
    _load_env_file,
    _nullify_erratic_values,
    do_sweep_replica,
    save_dict_to_json,
)
from pysdg.gen.tabsyn import (
    TabularDataset,
    sample,
    Decoder_model,
    TabSyn,
    pysdg_split_num_cat,
)
from pysdg.gen.tabddpm import (
    clava_clustering,
    clava_synthesizing,
    clava_training,
    load_configs,
    load_multi_table,
)
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

warnings.filterwarnings("ignore")

try:
    from ra_synthesis_sdk.model import stage
    from ra_synthesis_sdk.model.job import UtilityActionType
    from ra_synthesis_sdk.model.simulator_meta_info import SimulatorState
    from ra_synthesis_sdk.model.utility_assessment import UtilityAssessment
    from ra_synthesis_sdk.synthesis_client import SynthesisClient
    replica_exist = True
except ImportError:
    replica_exist = False


class Generator:

    def __init__(
        self,
        gen_name: str = "dummy",
        num_cores: int = None,
        work_dir: str = None,
        save_model=False,
        log_to_file: bool = False,
    ) -> None:
        """A class for synthetic data generation.

        Args:
            gen_name: The name of the generator. The default is 'dummy' which is used either to get the encoded version of the loaded dataframe using the 'load' method or to use the a pretrained model from pysdg vault zip file using the 'gen' method. If the generator is 'dummy' the rest of the arguments are ignored. The other available generators are:
            - 'replica/seq': Replica sequential generator (You need license to use this generator).
            - 'synthcity/bayesian_network': SynthCity Bayesian network generator.
            - 'synthcity/ctgan': SynthCity CTGAN generator.
            - 'synthcity/nflow': SynthCity Normalizing Flow generator.
            - 'synthcity/rtvae': SynthCity RTVAE generator.
            - 'synthcity/tvae': SynthCity TVAE generator.
            - 'synthcity/arf': SynthCity ARF generator.
            - 'yandex/tabddpm': Yandex TabDDPM generator.
            - 'amazon/tabsyn': Amazon TabSyn generator.
            num_cores: The maximum number of cores. If None, all available cores will be used. You may need to set that to a specific number for some generators such as the Bayesian network generator in SynthCity.where exploiting all available cores may cause memory issues.
            work_dir: User-defined work directory to which some processes may save temporary files and directories. Also, if 'save_model' is set to True,  the model artifact will be saved to this directory. If not provided, by default, this directory is created in the current working directory and it starts with 'pysdg'.
            save_model: A boolean whether or not to save the model artifact. If True, the model artifact will be saved to work_dir. Default is False.
            log_to_file: A boolean whether or not to save the log file. If True, the pysdg.log file will be saved to  work_dir. Default is False.
        """
        self.run_id = uuid.uuid4().hex
        self.naive_real = None
        self.real_info = None
        self.enc_real = None
        self.enc_real_info = None
        self.real_info_dtypes = None
        self.train_data_dtypes = None
        self.num_rows = None
        self.num_synths = None  # Initialize num_synths

        if gen_name == "dummy":
            self.work_dir = None
        else:
            if work_dir is None:
                self.work_dir = os.path.join(os.getcwd(), f"pysdgws{self.run_id}")
            else:
                self.work_dir = os.path.join(work_dir, f"pysdgws{self.run_id}")
            os.makedirs(self.work_dir, exist_ok=True)  # Ensure work_dir exists

        self.tmp_dir = None  # Initialize tmp_dir
        self.pysdg_vault_path = None  # Initialize pysdg_vault_path


        if not hasattr(Generator, "logger"):
            if log_to_file:
                log_filepath = os.path.join(self.work_dir, "pysdg.log")
                self.logger = configure_logging(log_filepath)
            else:
                self.logger = configure_logging()

        self.logger.info(
            "**************Started logging the generator: %s, num_cores= %s.**************",
            gen_name,
            num_cores,
        )

        valid_gen_names = [
            "replica_seq",
            "synthcity_bayesian_network",
            "synthcity_ctgan",
            "synthcity_nflow",
            "synthcity_rtvae",
            "synthcity_tvae",
            "synthcity_arf",
            "sdv_copula",
            "yandex_tabddpm",
            "amazon_tabsyn",
            "dummy",
        ]

        # Normalize gen_name to maintain downward compatibility
        gen_name = gen_name.replace("/", "_")

        # Map old names to new names
        name_mapping = {
            "replica_seq": "replica_seq",
            "replica/seq": "replica_seq",
            "synthcity_bayesian_network": "synthcity_bayesian_network",
            "synthcity/bayesian_network": "synthcity_bayesian_network",
            "synthcity_ctgan": "synthcity_ctgan",
            "synthcity/ctgan": "synthcity_ctgan",
            "synthcity_nflow": "synthcity_nflow",
            "synthcity/nflow": "synthcity_nflow",
            "synthcity_rtvae": "synthcity_rtvae",
            "synthcity/rtvae": "synthcity_rtvae",
            "synthcity_tvae": "synthcity_tvae",
            "synthcity/tvae": "synthcity_tvae",
            "synthcity_arf": "synthcity_arf",
            "synthcity/arf": "synthcity_arf",
            "sdv_copula": "sdv_copula",
            "sdv/copula": "sdv_copula",
            "yandex_tabddpm": "yandex_tabddpm",
            "yandex/tabddpm": "yandex_tabddpm",
            "amazon/tabsyn": "amazon_tabsyn",
            "dummy": "dummy",
        }

        # Apply name mapping
        gen_name = name_mapping.get(gen_name, gen_name)

        if gen_name not in valid_gen_names:
            self.logger.error(
                "Generator name '%s' is incorrect. Valid options are: %s",
                gen_name,
                ", ".join(valid_gen_names),
            )
            raise ValueError(
                f"Generator name '{gen_name}' is incorrect. Valid options are: {', '.join(valid_gen_names)}"
            )

        self.gen_name = gen_name
        self.num_cores = num_cores

        self.pysdg_artifacts = {}  # artifacts created by pysdg
        # artifacts created by source generator including its model(s). Both source and pysdg artifacts are saved in the same zip file (i.e pysdg_vault.zip).
        self.source_artifacts = None
        self.log_to_file = log_to_file
        self.save_model = save_model

        # If set to False, an error will be thrown if numeric variable (incl. cnt, dscrt and datetime) in the real dataset include non-numeric values. If set to True, the erratic value will be treated as missing value.
        self.suppress_errors = True
        self.synth_info = None
        self.synths = None
        self.enc_synths = None

        self.gen_params = {}

        # if "synthcity_" in gen_name:
        #     self.synthcity_params = self.gen_params

        if "replica_" in gen_name:
            if not replica_exist:
                self.logger.error(
                    "Replica generator library is not installed.")
                raise ImportError(
                    "Replica generator library is not installed. Please install it."
                )
            self.replica_client = self._connect_replica()
            self.replica_job_ids = []
            self.sweep_replica_jobs = False
            # self.tmp_dir = os.path.join(self.work_dir, "replica_tmp")
            # os.makedirs(self.tmp_dir, exist_ok=True)

        if "yandex_" in gen_name:
            self.tmp_dir = os.path.join(self.work_dir, "yandex_tmp")
            os.makedirs(self.tmp_dir, exist_ok=True)

        # if "amazon_" in gen_name:
        #     # self.work_dir = (
        #     #     os.path.join(os.getcwd(), f"pysdgws{self.run_id}")
        #     #     if work_dir is None
        #     #     else work_dir
        #     # )
        #     # os.makedirs(self.work_dir, exist_ok=True)

    def _enforce_json_types(self, df: pd.DataFrame, info: dict):

        for col_name in list(df.columns):
            try:
                if col_name in info["cat_names"]:
                    df[col_name] = df[col_name].astype("category")
                elif col_name in info["cnt_names"]:
                    df[col_name] = df[col_name].astype("float")
                elif col_name in info["dscrt_names"]:
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(
                        "Int64"
                    )
                    # df[col_name] = df[col_name].astype('int') #causes error with np.nan since it is float
                elif col_name in info["datetime_names"]:
                    df[col_name] = (
                        pd.to_datetime(df[col_name]).dt.tz_localize(
                            None).dt.normalize()
                    )
            except (ValueError, TypeError):
                df[col_name] = df[col_name].astype("str")
        return df

    def _validate_raw_info(self, raw_data: pd.DataFrame, raw_info: dict) -> None:
        """Validates the raw_info dictionary to ensure no duplicates in index lists.

        Args:
        raw_data (pd.DataFrame): The input data.
        raw_info (dict): The dictionary describing the real data.

        Raises:
        ValueError: If there are duplicates in the index lists.
        """

        # add quasi_idxs if it does not exist
        if "quasi_idxs" not in raw_info:
            raw_info["quasi_idxs"] = []

        # Check if all values in raw_info are lists
        single_item_keys = ["ds_name", "population_size", "h0_value"]
        for key in single_item_keys:
            if key in raw_info and not isinstance(raw_info[key], list):
                self.logger.warning(
                    "The value for '%s' should be provided as a list, even if it contains only a single element. This will become a requirement in future releases, and providing it as a single element will be deprecated.",
                    key
                )
                print(
                    f"The value for '{key}' should be provided as a list, even if it contains only a single element. This will become a requirement in future releases, and providing it as a single element will be deprecated."
                )
                raw_info[key] = [raw_info[key]]

        for key, value in raw_info.items():
            if not isinstance(value, list):
                self.logger.error(
                    "The value of '%s' in raw_info must be a list.", key)
                raise ValueError(
                    f"The value of '{key}' in raw_info must be a list.")

        # Check if the dataset name is a list containing exactly one element
        if not isinstance(raw_info["ds_name"], list) or len(raw_info["ds_name"]) != 1:
            self.logger.error(
                "raw_info['ds_name'] must be a list containing exactly one element."
            )
            raise ValueError(
                "raw_info['ds_name'] must be a list containing exactly one element."
            )

        # Check for missing type
        mandatory_fields = [
            "cat_idxs",
            "cnt_idxs",
            "dscrt_idxs",
            "datetime_idxs",
            "miss_vals",
        ]
        for field in mandatory_fields:
            if field not in raw_info:
                self.logger.error(
                    "Mandatory field '%s' is missing in raw_info.", field)
                raise ValueError(
                    f"Mandatory field '{field}' is missing in raw_info.")
            index_lists = ["cat_idxs", "cnt_idxs",
                           "dscrt_idxs", "datetime_idxs"]
            seen = set()

        # Check if all data column indexes are included in dic
        all_columns = set(range(raw_data.shape[1]))
        listed_columns = set(
            raw_info["cat_idxs"]
            + raw_info["cnt_idxs"]
            + raw_info["dscrt_idxs"]
            + raw_info["datetime_idxs"]
        )
        if not all_columns.issubset(listed_columns):
            missing_columns = all_columns - listed_columns
            self.logger.error(
                "The following columns are not listed in raw_info: %s", missing_columns
            )
            raise ValueError(
                f"The following columns are not listed in raw_info: {missing_columns}"
            )

        # Check for duplicate indexes
        for index_list in index_lists:
            for idx in raw_info[index_list]:
                if idx in seen:
                    self.logger.error(
                        "Duplicate index '%s' found in '%s'.", idx, index_list
                    )
                    raise ValueError(
                        f"Duplicate index '{idx}' found in '{index_list}'."
                    )
                seen.add(idx)
        self.logger.info(
            "Checking the input metadata for any conflict in variable indexes - Passed."
        )

    def _infer_dtypes(self, raw_data: pd.DataFrame):
        cat_names = raw_data.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        cnt_names = raw_data.select_dtypes(include=["float"]).columns.tolist()
        dscrt_names = raw_data.select_dtypes(include=["int"]).columns.tolist()
        datetime_names = raw_data.select_dtypes(
            include=["datetime"]).columns.tolist()

        # Include any additional types in categorical
        all_columns = set(raw_data.columns)
        identified_columns = set(
            cat_names + cnt_names + dscrt_names + datetime_names)
        additional_cat_idxs = list(all_columns - identified_columns)
        cat_names.extend(additional_cat_idxs)

        return list(cat_names), list(cnt_names), list(dscrt_names), list(datetime_names)

    def load(
        self, raw_data: Union[str, pd.DataFrame], raw_info: Union[str, dict] = None
    ) -> pd.DataFrame:
        """Safely loads the input dataframe and prepares it for training the generative model.

        Args:
            raw_data (str or pd.DataFrame):
                - If a string, it should the full path to the input real dataset csv file including extension.
                - If a pandas DataFrame, it should be the Pandas dataframe of the real data.
            raw_info (str or dict):
                - If a string, it should be the full path to the json file describing the real data including its extension.
                - If a dictionary, it should be the dictionary describing the real data.

        Returns:
            The loaded real data with all missing values properly processed.
        """

        if isinstance(raw_data, str):
            safe_data = pd.read_csv(raw_data, dtype=str, keep_default_na=False)
            if raw_info is None:
                raw_df = pd.read_csv(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            if hasattr(raw_data, "_rclass"):
                raw_data = raw_data.replace(-2147483648, np.NaN)
            if raw_info is None:
                raw_df = raw_data
            safe_data = raw_data.astype(str)
        else:
            self.logger.error(
                "raw_data must be either a file path (str) or a DataFrame"
            )
            raise ValueError(
                "raw_data must be either a file path (str) or a DataFrame")

        if isinstance(raw_info, str):
            with open(raw_info, "r", encoding="utf-8") as f:
                safe_info = json.load(f)
            self._validate_raw_info(safe_data, safe_info)
            safe_info = _get_var_names(safe_data, safe_info)
            safe_info["ds_fname"] = (
                Path(raw_data).stem + Path(raw_data).suffix
                if isinstance(raw_data, str)
                else None
            )  # this will be available only if the data is read from files
        elif isinstance(raw_info, dict):
            safe_info = raw_info
            self._validate_raw_info(safe_data, safe_info)
            safe_info = _get_var_names(safe_data, safe_info)
        elif raw_info is None:
            cat_names, cnt_names, dscrt_names, datetime_names = self._infer_dtypes(
                raw_df
            )
            safe_info = {
                "ds_name": ["unnamed"],
                "id_name": [],
                "cat_names": cat_names,
                "cnt_names": cnt_names,
                "dscrt_names": dscrt_names,
                "datetime_names": datetime_names,
                "text_names": [],
                "miss_vals": ["nan", "NAN", "<NA>", "NaT", "NaN", "NA", "None", ""],
                "h0_value": [0],
                "population_size": [10 * len(raw_df)],
                "quasi_names": list(raw_df.columns),
            }

        else:
            self.logger.error(
                "raw_info must be either a file path (str) or a dictionary"
            )
            raise ValueError(
                "raw_info must be either a file path (str) or a dictionary"
            )

        real = safe_data.replace(safe_info["miss_vals"], np.NaN)

        real = _nullify_erratic_values(
            safe_info, real, suppress_errors=self.suppress_errors
        )
        real = self._enforce_json_types(real, safe_info)
        enc_real_data, enc_real_info = _encode_missings(
            safe_data, safe_info, self.suppress_errors
        )  # Encode missing values in real data

        if "replica" in self.gen_name:
            for col in enc_real_data.select_dtypes(include=["category"]).columns:
                enc_real_data[col] = enc_real_data[col].astype(str)
        self.naive_real = safe_data
        self.real_info = safe_info
        self.enc_real = enc_real_data
        self.enc_real_info = enc_real_info
        self.real_info_dtypes = real.dtypes
        self.logger.info(
            "The dataset %s is loaded into the generator %s",
            safe_info['ds_name'],
            self.gen_name
        )

        self.pysdg_artifacts["ds_name"] = safe_info["ds_name"]
        self.pysdg_artifacts["gen_name"] = self.gen_name
        self.pysdg_artifacts["naive_real"] = self.naive_real
        self.pysdg_artifacts["enc_real"] = self.enc_real

        return real

    def restore_col_names(self, enc_df: pd.DataFrame) -> pd.DataFrame:
        """Curates the column names of the encoded dataframe.

        Args:
            enc_df: The encoded dataframe.

        Returns:
            The encoded dataframe with curated column names.
        """
        enc_df_copy = enc_df.copy()
        enc_df_copy = enc_df_copy.loc[
            :, [col for col in enc_df.columns if "_missing" not in col]
        ]
        col_names = enc_df_copy.columns
        new_col_names = []
        for col in col_names:
            new_col_name = col.split("%%%")[0]
            new_col_names.append(new_col_name)
        enc_df_copy.columns = new_col_names

        if self.gen_name == "dummy":  # the only usage of dummy if to get the encoded data and then restore the column names
            self._purge_work_dir()

        return enc_df_copy

    def _duplicate_rows(self, small_df, large_df):
        small_size = small_df.shape[0]
        large_size = large_df.shape[0]

        if small_size >= large_size:
            return small_df

        # Calculate the number of times to repeat the small dataframe
        repeat_times = large_size // small_size
        remainder = large_size % small_size

        # Duplicate rows
        duplicated_df = small_df
        for _ in range(repeat_times - 1):
            duplicated_df = duplicated_df.append(small_df, ignore_index=True)

        # If there's a remainder, add extra rows
        if remainder > 0:
            duplicated_df = duplicated_df.append(
                small_df.iloc[:remainder], ignore_index=True
            )

        return duplicated_df

    def _replace_duplicates(self, lst):
        count_dict = {}
        new_list = []
        for item in lst:
            if item in count_dict:
                count_dict[item] += 1
                new_item = f"{item}_{count_dict[item]}"
            else:
                count_dict[item] = 0
                new_item = item
            new_list.append(new_item)
        return new_list

    def unload(self) -> list:
        """Safely unloads the generated encoded synthetic dataset versions (synths).

        Returns:
            List of 'synths'. All 'synths' have the same number of records and matching variable types od 'real'.
        """
        enc_synth_lst = self.enc_synths
        impute = False  # if False (default), do not impute in the ouput

        syn_data_lst = []
        # if 'synthcity' in self.gen_name:
        for enc_syn_data in enc_synth_lst:
            if len(enc_syn_data) < len(self.naive_real):
                duplicated_enc_syn_data = self._duplicate_rows(
                    enc_syn_data, self.naive_real
                )  # increase the size of  enc_syn_data to match that of self.real_data (replicate rows in enc_syn_data)
                syn_data, syn_info = _decode_missings(
                    self.naive_real,
                    duplicated_enc_syn_data,
                    self.enc_real_info,
                    impute=impute,
                )
                syn_data = syn_data.iloc[
                    0: len(enc_syn_data), :
                ]  # restore the initial size of enc_syn_data by removing the added rows
            elif len(enc_syn_data) > len(self.naive_real):
                duplicated_real_data = self._duplicate_rows(
                    self.naive_real, enc_syn_data
                )  # increase the size of  real data to match that of enc_syn_data (by replicating rows)
                syn_data, syn_info = _decode_missings(
                    duplicated_real_data,
                    enc_syn_data,
                    self.enc_real_info,
                    impute=impute,
                )
                for id_col in self.real_info[
                    "id_name"
                ]:  # for all ID variables, replcae the duplicate values by adding a suffix to any duplication
                    syn_data[id_col] = self._replace_duplicates(
                        list(syn_data[id_col]))
            else:
                syn_data, syn_info = _decode_missings(
                    self.naive_real, enc_syn_data, self.enc_real_info, impute=impute
                )

            # if self.unify_na:
            syn_data.replace(syn_info["miss_vals"], np.NaN, inplace=True)
            syn_data = syn_data.fillna(np.NaN)
            syn_data = self._enforce_json_types(syn_data, self.real_info)
            syn_data_lst.append(syn_data)

        if "replica" in self.gen_name:
            if self.sweep_replica_jobs:  # clean replica files to keep storage.
                do_sweep_replica(self.replica_client, self.replica_job_ids)

        if "yandex" in self.gen_name:
            dir_to_delete = self.tmp_dir
            self._make_writable_and_remove(dir_to_delete)


        # if self.delete_work_dir:
        #     self._make_writable_and_remove(self.work_dir)

        self.synths = syn_data_lst
        self.synth_info = syn_info

        # if (
        #     self.save_model is False
        #     and self.log_to_file is False
        #     and self.work_dir is not None
        # ):
        # Only purge work_dir if it is empty
        if self.work_dir and os.path.isdir(self.work_dir) and not os.listdir(self.work_dir):
            self._purge_work_dir()

        return syn_data_lst

    def _select_gen4train(self, train_data: pd.DataFrame) -> None:
        if "replica_" in self.gen_name:
            artifact = self._train4replica(train_data)
            self.source_artifacts = artifact

        elif "synthcity_" in self.gen_name:
            if any(x in self.gen_name for x in ["_ctgan", "_tvae", "_rtvae"]):
                model = self._train4synthcity_w_fallback(
                    train_data, min_batch_size=10)
                self.source_artifacts = model
            else:
                model = self._train4synthcity(train_data)
                self.source_artifacts = model

        elif "sdv_" in self.gen_name:
            model = self._train4sdv(train_data)
            self.source_artifacts = model

        elif "yandex_" in self.gen_name:
            if torch.cuda.is_available():
                pysdg_device_name = "cuda"
                self.logger.info(
                    "CUDA device is available. Using GPU for training.")
            else:
                pysdg_device_name = "cpu"
                self.logger.warning(
                    "CUDA device is not available. Using CPU for training. This will be extremely slow."
                )
            yandex_artifact = self._train4yandex(train_data, pysdg_device_name)
            self.source_artifacts = yandex_artifact

        elif "amazon_" in self.gen_name:
            if torch.cuda.is_available():
                pysdg_device_name = "cuda"
                self.logger.info(
                    "CUDA device is available. Using GPU for training.")
            else:
                pysdg_device_name = "cpu"
                self.logger.warning(
                    "CUDA device is not available. Using CPU for training. This will be extremely slow."
                )
            amazon_artifact = self._train4amazon(train_data, pysdg_device_name)
            self.source_artifacts = amazon_artifact
        else:
            raise ValueError("Generator name is incorrect.")

    def train(self, input_data: pd.DataFrame = None) -> None:
        """Trains the generator using the encoded real or the input dataset. To avoid training errors, make sure to use the load method first.

        Args:
            input_data: A pandas dataframe that is used to train the model. If passed, it should be a subset of encoded real dataset.
        """

        if not isinstance(self.enc_real, pd.DataFrame):
            self.logger.error("You cannot train before loading data.")
            raise ValueError("You cannot train before loading data.")

        if input_data is None:
            train_data = self.enc_real
        else:
            is_subset = (
                input_data.isin(self.enc_real.to_dict(
                    orient="list")).all().all()
            )
            if not is_subset:
                self.logger.error(
                    "input_data is not a subset of self.enc_real")
                raise ValueError("input_data is not a subset of self.enc_real")
            train_data = input_data

        if self.num_cores is not None:
            with threadpool_limits(limits=self.num_cores):
                self._select_gen4train(train_data)
        else:
            self._select_gen4train(train_data)

        self.pysdg_artifacts["real_info"] = self.real_info
        self.pysdg_artifacts["enc_real_info"] = self.enc_real_info
        self.train_data_dtypes = train_data.dtypes
        self.pysdg_artifacts["train_data_dtypes"] = train_data.dtypes

        if self.save_model:
            try:
                zip_path = os.path.join(self.work_dir, "pysdg_vault.zip")
                self._save_to_zip(self.pysdg_artifacts,
                                  self.source_artifacts, zip_path)
                self.logger.info(
                    "Saved artifcat and model as a vault to %s.", zip_path)
            except (RuntimeError, ValueError, KeyError) as e:
                self.logger.error(
                    "Failed to save the artifact and model: %s", e)

    def gen(
        self,
        num_rows: int = None,
        num_synths: int = None,
        pysdg_vault_path: str = None,
    ) -> list:
        """Generates multiple synthetic datasets (synths) from the trained generative model.

        Args:
            num_rows: The target number of required records (observations) in the output synthetic data.
            num_synths: The target number of required synthetic versions (synths) where each 'synth' has the same number of the target num_obsv.
            pysdg_vault_path: The path to the pysdg vault zip file. If provided, it will load the model from this path instead of using the trained model in memory.

        """

        # If a pretrained model is provided, gen_name must be 'dummy'
        if pysdg_vault_path is not None and self.gen_name != "dummy":
            self.logger.error(
                "If a pretrained model is provided, please define your generator object without inputting any gen_name."
            )
            raise ValueError(
                "If a pretrained model is provided, please define your generator object without inputting any gen_name."
            )

        if num_rows is None or num_synths is None:  # if not provided, use the number of rows in the real dataset
            if self.enc_real is None:
                self.logger.error(
                    "You cannot generate synthetic data before training the model."
                )
                raise ValueError(
                    "You cannot generate synthetic data before training the model."
                )
            self.num_rows = len(self.enc_real)
            self.num_synths = 1

        else:
            self.num_rows = num_rows
            self.num_synths = num_synths

        if pysdg_vault_path is not None:
            if not pysdg_vault_path.endswith(".zip"):
                self.logger.error(
                    "The provided pysdg_vault_path does not point to a .zip file."
                )
                raise ValueError(
                    "The provided pysdg_vault_path does not point to a .zip file."
                )
            
            self.pysdg_vault_path = pysdg_vault_path

            self.work_dir = os.path.join(os.path.dirname(os.path.abspath(pysdg_vault_path)), f"pysdgws{self.run_id}")

            self.pysdg_artifacts, self.source_artifacts = self._load_from_zip(
                pysdg_vault_path
            )
            self.train_data_dtypes = self.pysdg_artifacts["train_data_dtypes"]
            self.enc_real_info = self.pysdg_artifacts["enc_real_info"]
            self.real_info = self.pysdg_artifacts["real_info"]
            self.naive_real = self.pysdg_artifacts["naive_real"]
            self.enc_real = self.pysdg_artifacts["enc_real"]
            self.gen_name = self.pysdg_artifacts["gen_name"]
            self.logger.info(
                "Loaded pysdg vault from %s. The generator name is %s.",
                pysdg_vault_path,
                self.gen_name,
            )

            if "replica_" in self.gen_name:
                if not replica_exist:
                    self.logger.error(
                        "Replica generator library is not installed.")
                    raise ImportError(
                        "Replica generator library is not installed. Please install it."
                    )
                self.replica_client = self._connect_replica()
                self.replica_job_ids = []
                self.sweep_replica_jobs = True

                self.tmp_dir = os.path.join(self.work_dir, "replica_tmp")
                os.makedirs(self.tmp_dir, exist_ok=True)

            if "yandex_" in self.gen_name:
                self.tmp_dir = os.path.join(self.work_dir, "yandex_tmp")
                os.makedirs(self.tmp_dir, exist_ok=True)


        if not hasattr(self, "pysdg_artifacts") or self.pysdg_artifacts is None:
            self.logger.error(
                "No model exists. Please call self.train() first")
            raise ValueError("No model exists. Please call self.train() first")
        if "replica_" in self.gen_name:
            enc_synths_raw = self._gen4replica(
                self.source_artifacts, self.num_rows, self.num_synths)
        elif "synthcity_" in self.gen_name:
            enc_synths_raw = self._gen4synthcity(
                self.source_artifacts, self.num_rows, self.num_synths)
        elif "sdv_" in self.gen_name:
            enc_synths_raw = self._gen4sdv(
                self.source_artifacts, self.num_rows, self.num_synths)
        elif "yandex_" in self.gen_name:
            enc_synths_raw = self._gen4yandex(
                self.source_artifacts, self.num_rows, self.num_synths)
        elif "amazon_" in self.gen_name:
            enc_synths_raw = self._gen4amazon(
                self.source_artifacts, self.num_rows, self.num_synths)
        else:
            raise ValueError("Generator name is incorrect..")
        enc_synths = []
        for enc_synth_raw in enc_synths_raw:
            non_logical_columns = [
                col for col in enc_synth_raw.columns if "_missing" not in col
            ]
            # non_logical_columns = enc_synth_raw.columns[
            #     enc_synth_raw.dtypes != "bool"
            # ].tolist()
            enc_synth_raw[non_logical_columns] = enc_synth_raw[
                non_logical_columns
            ].astype("str")
            enc_synth = enc_synth_raw.astype(self.train_data_dtypes)
            enc_synths.append(enc_synth)
        self.enc_synths = enc_synths

    # REPLICA SEQ GENERATOR

    def _connect_replica(self) -> object:  # returns replica client object
        _load_env_file()  # get credentials
        DEFINED_CLIENT_ID = os.getenv("REPLICA_ID")
        DEFINED_CLIENT_SECRET = os.getenv("REPLICA_PWD")
        # HTTP APIs access
        BASE_HOST = os.getenv(
            "REPLICA_HOST"
        )  # private IP of Synthesis Core VM - get it from AWS Console or GCP console
        BASE_PORT = os.getenv("REPLICA_PORT")
        client = SynthesisClient(host=BASE_HOST, port=BASE_PORT)
        client.timeout_seconds = 360000
        client.set_credentials(
            client_id=DEFINED_CLIENT_ID, client_secret=DEFINED_CLIENT_SECRET
        )
        if not client.login(with_scopes=client.USER_SCOPES):
            self.logger.error(
                "Login failed. Please ensure your .env file has the correct credentials."
            )
            raise ValueError(
                "Login failed. Please ensure your .env file has the correct credentials."
            )
        return client

    def _train4replica(self, train_data):
        client = self.replica_client
        job_id = client.create_job()
        self.replica_job_ids.append(job_id)
        self.logger.info("Replica job %s created..", job_id)
        temp_dir = os.path.join(self.work_dir, "temp4replica")
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4().hex}.csv"
        tmp_real_path = os.path.join(temp_dir, unique_filename)
        train_data.to_csv(tmp_real_path, index=False)
        if os.path.isfile(tmp_real_path):
            client.add_input(job_id, tmp_real_path)
            client.generate_synthesis_plan(job_id)
            client.wait_for_not_running(job_id)
            assert client.check_if_valid(job_id)
            assert client.preprocess(job_id, wait_time=3600000)  # 60000
            assert client.synthesize(
                job_id, num_recs=len(train_data), wait_time=3600000
            )
            assert client.postprocess(job_id, wait_time=3600000)  # 60000
            self.logger.info("Training using Replica is done!")
            SimulatorMetaInfo = client.create_simulator(
                job_id,
                sim_name="per_step_replica_model",
                sim_desc="Replica Simulator for a specific step",
                model_only=True,
            )
        else:
            raise FileNotFoundError("Temporary file does not exist!!")
        shutil.rmtree(temp_dir)
        self.replica_job_ids.append(SimulatorMetaInfo.id)
        return SimulatorMetaInfo.id

    def _gen4replica(self, simulator_id, num_rows, num_synths):
        client = self.replica_client
        syn_synths = []
        for i in range(num_synths):
            try:
                timeout = 3600000
                mustend = time.time() + timeout
                while time.time() < mustend:
                    SimulatorMetaInfo = client.get_simulator(simulator_id)
                    if SimulatorMetaInfo.state is SimulatorState.READY:
                        break
                for j in range(10):  # try 10 times to get simulator_synthesis_job
                    try:  # try twice
                        simulator_synthesis_job = client.synthesize_from_simulator(
                            sim_id=simulator_id,
                            num_recs=num_rows,
                            prefix=True,
                            generate_preview=False,
                        )
                        self.replica_job_ids.append(simulator_synthesis_job.id)
                        break
                    except:
                        if j == 9:
                            raise Exception(
                                "Simulator synthesis job NOT available!!")
                        continue
                for m in range(
                    10
                ):  # try 10 times to download the zipped file (that includes the synth csv) of replica
                    try:
                        syn_result_path = client.download_synthetic(
                            simulator_synthesis_job.id, self.work_dir
                        )
                        if os.path.isfile(syn_result_path):
                            break
                    except:
                        if m == 9:
                            raise Exception(
                                "Simulator failed to save synth !!")
                generated_data = pd.read_csv(
                    syn_result_path, compression="zip")
                if not isinstance(
                    generated_data, pd.DataFrame
                ):  # Make sure generated_data is a dataframe
                    raise Exception("Saved synth is not pandas dataframe !!")
                else:
                    # generated_data=generated_data.astype('str')
                    syn_synths.append(generated_data)
                    os.remove(syn_result_path)
                    self.logger.info(
                        "-- - Replica generating synth no. %s of size %s -- Completed!", i, generated_data.shape
                    )
            except Exception as e:
                self.logger.error(
                    "- Failed to generate synth no. %s -- ☹️ -- %s", i, e)
                continue
        return syn_synths

    # SYNTHCITY

    def _get_gen_name4synthcity(self):
        gen_name4synthcity = self.gen_name.split("_")[
            1:
        ]  # drop synthcity from the name of the generator
        gen_name4synthcity = (
            "_".join(gen_name4synthcity)
            if len(gen_name4synthcity) > 1
            else gen_name4synthcity[0]
        )
        #self.gen_name4synthcity = gen_name4synthcity
        return gen_name4synthcity

    def _train4synthcity(self, transformed_in_df):
        gen_name4synthicty = self._get_gen_name4synthcity()
        loader = GenericDataLoader(transformed_in_df)
        syn_model = Plugins().get(gen_name4synthicty, **self.gen_params)
        # Train
        self.logger.info("Started training using %s...", self.gen_name)
        syn_model.fit(loader)
        assert len(syn_model.data_info["static_features"]) == len(
            transformed_in_df.columns
        ), "Model failed to capture all input features"
        self.logger.info("Completed training using %s.", self.gen_name)
        return syn_model

    def _train4synthcity_w_fallback(self, transformed_in_df, min_batch_size):
        gen_name4synthicty = self._get_gen_name4synthcity()
        if not "batch_size" in self.gen_params:
            syn_model = Plugins().get(gen_name4synthicty)
            self.gen_params["batch_size"] = (
                syn_model.batch_size
            )  # retrieve the default batch size

        loader = GenericDataLoader(transformed_in_df)
        self.logger.info("Started training using %s...", self.gen_name)
        while self.gen_params["batch_size"] >= min_batch_size:
            try:
                syn_model = Plugins().get(gen_name4synthicty, **self.gen_params)
                self.logger.info(
                    "No of Iterations=%s, Batch Size=%s", syn_model.n_iter, syn_model.batch_size
                )
                # Train
                syn_model.fit(loader)
                assert len(syn_model.data_info["static_features"]) == len(
                    transformed_in_df.columns
                ), "Model failed to capture all input features"
                self.logger.info("Completed training using %s.", self.gen_name)
                return syn_model
            except RuntimeError as e:
                self.logger.warning(
                    "Unable to train with batch size %s: %s - Trying to reduce the batch size..",
                    self.gen_params['batch_size'], e
                )
                self.gen_params["batch_size"] = self.gen_params["batch_size"] // 2
                if self.gen_params["batch_size"] < min_batch_size:
                    self.logger.error(
                        "Minimum batch size reached. Unable to train the model. Possibly failure is caused by the heavily imbalanced categories in one of the variables."
                    )
                    raise ValueError("Training failed.") from e

    def _gen4synthcity(self, syn_model, num_rows, num_synths):
        syn_synths = []
        i = 0
        for i in range(num_synths):
            try:
                syn_synth = self._gen_one_synth4synthcity(
                    syn_model, num_rows)
                if not isinstance(syn_synth, pd.DataFrame):
                    self.logger.error(
                        "Generator did not succeed in generating a synth"
                    )
                    raise RuntimeError(
                        "Generator did not succeed in generating a synth")
                self.logger.info(
                    "Generating synth no. %s of size %s -- Completed!", i, syn_synth.shape
                )
                syn_synths.append(syn_synth)
            except (RuntimeError, ValueError) as e:
                self.logger.error(
                    "Failed to generate synth no. %s -- ☹️ -- %s", i, e)
                continue
        return syn_synths

    def _gen_one_synth4synthcity(self, syn_model, target_no_obsv: int) -> pd.DataFrame:
        one_synth = []
        secured_no_obsv = 0
        divisor = 1
        # test_logic=0
        trials_count = 0
        possible_no_obsv = target_no_obsv
        while (secured_no_obsv < target_no_obsv) and (
            trials_count < 500
        ):  # keep trying upto 500 times
            trials_count += 1
            try:
                # if test_logic:
                possible_synth_df = syn_model.generate(
                    count=possible_no_obsv
                ).dataframe()
                # else:
                # raise Exception("Deliberate Error")
                one_synth.append(possible_synth_df)
                secured_no_obsv += possible_no_obsv
                possible_no_obsv = target_no_obsv - secured_no_obsv
                divisor = 1
            except Exception:
                divisor += 1
                possible_no_obsv = (
                    target_no_obsv - secured_no_obsv) // divisor
                self.logger.warning(
                    "******* Reducing number of observations to %s and re-trying..", possible_no_obsv
                )
                continue
        if trials_count >= 500:
            self.logger.error("No synth returned after 500 attempts!")
        one_synth_df = pd.concat(one_synth, ignore_index=True)
        one_synth_df = one_synth_df.sample(frac=1).reset_index(drop=True)
        return one_synth_df

    # SDV

    def _train4sdv(self, transformed_in_df):
        gen_name4sdv = self.gen_name.split("_")[1]
        metadata = Metadata.detect_from_dataframe(
            data=transformed_in_df, table_name=f'enc_{self.enc_real_info["ds_name"][0]}'
        )
        for col in transformed_in_df.columns:
            dtype = transformed_in_df[col].dtype
            if pd.api.types.is_categorical_dtype(dtype) or dtype == bool:
                metadata.update_column(column_name=col, sdtype="categorical")
            else:
                metadata.update_column(column_name=col, sdtype="numerical")

        if gen_name4sdv == "copula":
            model = GaussianCopulaSynthesizer(metadata)
        else:
            self.logger.error("Generator name is incorrect.")
            raise ValueError("Generator name is incorrect.")
        self.logger.info("Started training using %s...", self.gen_name)
        model.fit(transformed_in_df)
        self.logger.info("Completed training using %s.", self.gen_name)
        return model

    def _gen4sdv(self, syn_model, num_rows, num_synths):
        syn_synths = []
        i = 0
        for i in range(num_synths):
            try:
                syn_synth = self._gen_one_synth4sdv(syn_model, num_rows)
                if not isinstance(syn_synth, pd.DataFrame):
                    raise RuntimeError(
                        "Generator did not succeed in generating a synth")
                self.logger.info(
                    "Generating synth no. %s of size %s -- Completed!", i, syn_synth.shape
                )
                syn_synths.append(syn_synth)
            except (RuntimeError, ValueError) as e:
                self.logger.error(
                    "Failed to generate synth no. %s -- ☹️ -- %s", i, e)
                continue
        return syn_synths

    def _gen_one_synth4sdv(self, syn_model, target_no_obsv: int) -> pd.DataFrame:
        one_synth = []
        secured_no_obsv = 0
        divisor = 1
        # test_logic=0
        trials_count = 0
        possible_no_obsv = target_no_obsv
        while (secured_no_obsv < target_no_obsv) and (trials_count < 500):
            trials_count += 1
            try:
                possible_synth_df = syn_model.sample(target_no_obsv)
                one_synth.append(possible_synth_df)
                secured_no_obsv += possible_no_obsv
                possible_no_obsv = target_no_obsv - secured_no_obsv
                divisor = 1
            except:
                divisor += 1
                possible_no_obsv = (
                    target_no_obsv - secured_no_obsv) // divisor
                self.logger.warning(
                    "******* Reducing number of observations to %s and re-trying..", possible_no_obsv
                )
                continue
        if trials_count >= 500:
            self.logger.error("No synth returned after 500 attempts!")
        one_synth_df = pd.concat(one_synth, ignore_index=True)
        one_synth_df = one_synth_df.sample(frac=1).reset_index(drop=True)
        return one_synth_df

    # YANDEX TABDDPM

    def _convert_ds_info2yandex_domain(self, dic, df):

        # Initialize the output dictionary
        dic_out = {}

        # Process categorical variables
        for var in dic.get(
            "cat_names", []
        ):  # Safely get 'cat_names' from the dictionary
            if var not in dic.get("dropped_names", []):
                dic_out[var] = {
                    # Count unique values in the column
                    "size": df[var].nunique(),
                    "type": "discrete",
                }

        # Process continuous variables
        for var in dic.get(
            "cnt_names", []
        ):  # Safely get 'cnt_names' from the dictionary
            if var not in dic.get("dropped_names", []):
                dic_out[var] = {
                    # Count unique values in the column
                    "size": df[var].nunique(),
                    "type": "continuous",
                }

        # Process discrete variables
        for var in dic.get(
            "dscrt_names", []
        ):  # Safely get 'dscrt_names' from the dictionary
            if var not in dic.get("dropped_names", []):
                dic_out[var] = {
                    # Count unique values in the column
                    "size": df[var].nunique(),
                    "type": "continuous",
                }

        # Process datetime variables
        for var in dic.get(
            "datetime_names", []
        ):  # Safely get 'dscrt_names' from the dictionary
            if var not in dic.get("dropped_names", []):
                dic_out[var] = {
                    # Count unique values in the column
                    "size": df[var].nunique(),
                    "type": "continuous",
                }

        # Process addional variables tracking missing values
        for var in dic.get(
            "num_names_w_miss", []
        ):  # Safely get 'dscrt_names' from the dictionary
            if var not in dic.get("dropped_names", []):
                dic_out[f"{var}_missing"] = {
                    "size": df[
                        f"{var}_missing"
                    ].nunique(),  # Count unique values in the column
                    "type": "discrete",
                }

        return dic_out

    def _make_configs4yandex(
        self,
        transformed_in_df: pd.DataFrame,
        tmp_dir: str,
    ):
        mandatory_data_fname = "train.csv"
        tmp_real_path = os.path.join(tmp_dir, mandatory_data_fname)
        # Convert boolean columns to integer
        for col in transformed_in_df.select_dtypes(include=["bool"]).columns:
            transformed_in_df[col] = transformed_in_df[col].astype(int)
        transformed_in_df.to_csv(tmp_real_path, index=False)
        ds_name = self.enc_real_info["ds_name"][0]

        # Step 1: Build {ds_name}.json file from
        basic_config = {
            "general": {
                "data_dir": tmp_dir,
                "exp_name": ds_name,
                "workspace_dir": tmp_dir,
                "sample_prefix": "",
                "test_data_dir": tmp_dir,
            }
        }

        default_params = {
            "clustering": {
                "parent_scale": 1.0,
                "num_clusters": 50,
                "clustering_method": "both",
            },
            "diffusion": {
                "d_layers": [512, 1024, 1024, 1024, 1024, 512],
                "dropout": 0.0,
                "num_timesteps": 200,
                "model_type": "mlp",
                "iterations": 10000,
                "batch_size": 4096,
                "lr": 0.0006,
                "gaussian_loss_type": "mse",
                "weight_decay": 1e-05,
                "scheduler": "cosine",
            },
            "classifier": {
                "d_layers": [128, 256, 512, 1024, 512, 256, 128],
                "lr": 0.0001,
                "dim_t": 128,
                "batch_size": 4096,
                "iterations": 2000,
            },
            "sampling": {"batch_size": 1000, "classifier_scale": 1.0},
            "matching": {
                "num_matching_clusters": 1,
                "matching_batch_size": 100,
                "unique_matching": True,
                "no_matching": False,
            },
        }

        custom_params = self._merge_dicts(default_params, self.gen_params)

        self.gen_params = custom_params

        dictionary = {**basic_config, **custom_params}

        # Step 2: Write the dictionary to a JSON file
        output_file1 = os.path.join(tmp_dir, f"{ds_name}.json")
        with open(output_file1, "w") as f:
            json.dump(dictionary, f, indent=4)

        # Step 3: Build the dataset_meta.json from scratch
        dictionary = {
            "relation_order": [[None, ds_name]],
            "tables": {ds_name: {"children": [], "parents": []}},
        }

        # Step 4: Write the dataset_meta.json to a JSON file
        output_file2 = os.path.join(tmp_dir, "dataset_meta.json")
        with open(output_file2, "w") as f:
            json.dump(dictionary, f, indent=4)

        # Step 5: Build the dataset {ds_name}_domain.json \
        domain_dict = self._convert_ds_info2yandex_domain(
            self.enc_real_info, transformed_in_df
        )
        # Step 6: Write the dataset {ds_name}_domain.json to a JSON file
        output_file3 = os.path.join(tmp_dir, f"{ds_name}_domain.json")
        with open(output_file3, "w") as f:
            json.dump(domain_dict, f, indent=4)

        return output_file1

    def _train4yandex(self, transformed_in_df, pysdg_device_name):
        if not self.gen_name == "yandex_tabddpm":
            self.logger.error("Generator name is incorrect.")
            raise ValueError("Generator name is incorrect.")
        tmp_dir = self.tmp_dir
        self.logger.info(f"Started training using {self.gen_name}...")
        config_path = self._make_configs4yandex(transformed_in_df, tmp_dir)
        configs, save_dir = load_configs(config_path)
        tables, relation_order, dataset_meta = load_multi_table(
            configs["general"]["data_dir"]
        )
        tables, all_group_lengths_prob_dicts = clava_clustering(
            tables, relation_order, save_dir, configs
        )
        models = clava_training(
            tables, relation_order, save_dir, configs, pysdg_device_name
        )  # we are training here and the model will be saved in the save dir

        self.logger.info(f"Completed training using {self.gen_name}.")
        yandex_artifact = {
            "tables": tables,
            "relation_order": relation_order,
            "save_dir": save_dir,
            "all_group_lengths_prob_dicts": all_group_lengths_prob_dicts,
            "models": models,
            "configs": configs,
        }
        return yandex_artifact

    def _gen_one_synth4yandex(self, yandex_artifact, num_rows: int) -> pd.DataFrame:
        tables = yandex_artifact["tables"]
        configs = yandex_artifact["configs"]
        ds_name = self.enc_real_info["ds_name"][0]
        # sample_scale = (1 if "debug" not in configs else configs["debug"]["sample_scale"])  # original implementation
        sample_scale = num_rows/len(tables[ds_name]['df'])
        if self.pysdg_vault_path:
            last_path=os.path.basename(yandex_artifact["save_dir"])
            yandex_artifact["save_dir"]=os.path.join(self.tmp_dir, last_path)
            before_matching_dir = os.path.join(yandex_artifact["save_dir"], "before_matching")
            os.makedirs(before_matching_dir, exist_ok=True)
            configs["general"]["workspace_dir"] = self.tmp_dir
  
        cleaned_tables, synthesizing_time_spent, matching_time_spent = (
            clava_synthesizing(
                tables,
                yandex_artifact["relation_order"],
                yandex_artifact["save_dir"],
                yandex_artifact["all_group_lengths_prob_dicts"],
                yandex_artifact["models"],
                configs,
                sample_scale,
            )
        )

        return cleaned_tables[ds_name]

    def _gen4yandex(self, yandex_artifact, num_rows, num_synths):
        syn_synths = []
        i = 0
        # seed = 0  # default seed
        for i in range(num_synths):
            try:
                syn_synth = self._gen_one_synth4yandex(
                    yandex_artifact, num_rows)
                if not isinstance(syn_synth, pd.DataFrame):
                    raise RuntimeError(
                        "Generator did not succeed in generating a synth")
                self.logger.info(
                    "Generating synth no. %s of size %s -- Completed!", i, syn_synth.shape
                )
                syn_synths.append(syn_synth)    
            except (RuntimeError, ValueError) as e:
                self.logger.error(
                    "Failed to generate synth no. %s -- ☹️ -- %s", i, e)
                continue
        return syn_synths

    # amazon/tabsyn

    def _make_configs4amazon(self, transformed_in_df: pd.DataFrame, work_dir: str):

        default_params = {
            "test_size_ratio": 0.1,
            "task_type": "regression",
            "model_params": {"n_head": 1, "factor": 32, "num_layers": 2, "d_token": 4},
            "transforms": {
                # Literal["__none__", "standard", "quantile", "minmax"],
                "normalization": "quantile",
                "num_nan_policy": "mean",
                "cat_nan_policy": "__none__",
                "cat_min_frequency": "__none__",
                # Literal["__none__", "one-hot", "counter","label"],
                "cat_encoding": "label",
                "y_policy": "default",
            },
            "train": {
                "vae": {
                    "num_epochs": 4000,
                    "batch_size": 4096,
                    "num_dataset_workers": 4,
                },
                "diffusion": {
                    "num_epochs": 10001,
                    "batch_size": 4096,
                    "num_dataset_workers": 4,
                },
                "optim": {
                    "vae": {
                        "lr": 1e-3,
                        "weight_decay": 0,
                        "factor": 0.95,
                        "patience": 10,
                    },
                    "diffusion": {
                        "lr": 1e-3,
                        "weight_decay": 0,
                        "factor": 0.9,
                        "patience": 20,
                    },
                },
            },
            "loss_params": {"max_beta": 1e-2, "min_beta": 1e-5, "lambd": 0.7},
        }

        custom_params = self._merge_dicts(default_params, self.gen_params)
        self.gen_params = custom_params
        save_path = os.path.join(
            work_dir, "configs", f"{self.real_info['ds_name'][0]}.json"
        )
        save_dict_to_json(custom_params, save_path)

        return save_path

    def _get_numpy4amazon(self, transformed_in_df: pd.DataFrame):  # SMK
        """Converts the given DataFrame into numpy arrays and a mapping dictionary.

        Args:
            transformed_in_df (pd.DataFrame): The DataFrame to convert.

        Returns:
            A tuple containing two tuples of numpy arrays (X_train_num, X_test_num) and (X_train_cat, X_test_cat),
                   a mapping dictionary, processing objects for numerical and categorical data, lists of numeric and categorical column indices.
        """

        def process_dataframe(df: pd.DataFrame):
            # Step 1: Convert bool columns to category
            df = df.copy()
            for col in df.select_dtypes(include="bool").columns:
                df[col] = df[col].astype("category")

            # Step 2: Separate columns by type
            cat_cols = df.select_dtypes(include="category").columns
            num_cols = df.select_dtypes(include="float").columns

            num_col_idx = df.columns.get_indexer(num_cols)
            cat_col_idx = df.columns.get_indexer(cat_cols)

            X_cat = df[cat_cols].to_numpy()
            X_num = df[num_cols].to_numpy()

            # Step 3: Create mapping dictionary
            mapping_dict = {}
            for i, col in enumerate(num_cols):
                original_idx = df.columns.get_loc(col)
                mapping_dict[i] = original_idx
            offset = len(num_cols)
            for i, col in enumerate(cat_cols):
                original_idx = df.columns.get_loc(col)
                mapping_dict[offset + i] = original_idx

            return X_num, X_cat, mapping_dict, num_col_idx, cat_col_idx

        def transform_data(
            X_num: np.ndarray,
            X_cat: np.ndarray,
            cat_encoding: str = "none",
            num_normalizing: str = "none",
        ):
            processors_dict = {}

            # ----- Encode categorical data -----
            if cat_encoding == "one-hot":
                ohe = OneHotEncoder(sparse_output=False,
                                    handle_unknown="ignore")
                X_cat_encoded = ohe.fit_transform(X_cat)
                processors_dict["cat"] = ohe
            elif cat_encoding == "counter":
                X_cat_encoded = np.zeros_like(X_cat, dtype=float)
                counters = []
                for col in range(X_cat.shape[1]):
                    values, counts = np.unique(
                        X_cat[:, col], return_counts=True)
                    freq = dict(zip(values, counts))
                    X_cat_encoded[:, col] = [freq[val]
                                             for val in X_cat[:, col]]
                    counters.append(freq)
                processors_dict["cat"] = counters
            elif cat_encoding == "label":
                X_cat_encoded = np.zeros_like(X_cat, dtype=int)
                label_encoders = []
                for col in range(X_cat.shape[1]):
                    le = LabelEncoder()
                    X_cat_encoded[:, col] = le.fit_transform(X_cat[:, col])
                    label_encoders.append(le)
                processors_dict["cat"] = label_encoders
            elif cat_encoding == "__none__":
                X_cat_encoded = X_cat.copy()
                processors_dict["cat"] = None
            else:
                raise ValueError(f"Unknown cat_encoding: {cat_encoding}")

            # ----- Normalize numeric data -----
            if num_normalizing == "standard":
                scaler = StandardScaler()
                X_num_encoded = scaler.fit_transform(X_num)
                processors_dict["num"] = scaler
            elif num_normalizing == "quantile":
                quantile = QuantileTransformer(output_distribution="uniform")
                X_num_encoded = quantile.fit_transform(X_num)
                processors_dict["num"] = quantile
            elif num_normalizing == "minmax":
                minmax = MinMaxScaler()
                X_num_encoded = minmax.fit_transform(X_num)
                processors_dict["num"] = minmax
            elif num_normalizing == "__none__":
                X_num_encoded = X_num.copy()
                processors_dict["num"] = None
            else:
                raise ValueError(f"Unknown num_normalizing: {num_normalizing}")

            return X_num_encoded, X_cat_encoded, processors_dict

        def split_encoded_data(
            X_num: np.ndarray,
            X_cat: np.ndarray,
            split_ratio: float = 0.2,
            random_state: int = 42,
        ):
            X_train_num, X_test_num = train_test_split(
                X_num, test_size=split_ratio, random_state=random_state
            )
            X_train_cat, X_test_cat = train_test_split(
                X_cat, test_size=split_ratio, random_state=random_state
            )

            return (X_train_num, X_test_num), (X_train_cat, X_test_cat)

        X_num, X_cat, mapping_dict, num_col_idx, cat_col_idx = process_dataframe(
            transformed_in_df
        )
        X_num, X_cat, processors_dict = transform_data(
            X_num,
            X_cat,
            cat_encoding=self.gen_params["transforms"]["cat_encoding"],
            num_normalizing=self.gen_params["transforms"]["normalization"],
        )
        (X_train_num, X_test_num), (X_train_cat, X_test_cat) = split_encoded_data(
            X_num,
            X_cat,
            split_ratio=self.gen_params["test_size_ratio"],
            random_state=42,
        )

        return (
            (X_train_num, X_test_num),
            (X_train_cat, X_test_cat),
            mapping_dict,
            processors_dict,
            num_col_idx,
            cat_col_idx,
        )

    def _train4amazon(
        self,
        transformed_in_df: pd.DataFrame,
        pysdg_device_name: str,
        chkpnt_dir: str = None,
    ):
        """Trains the Amazon Tabsyn model using the provided training data.

        Args:
            train_data (pd.DataFrame): The training data.

        Returns:
            object: The trained Amazon Tabsyn model.
        """
        # Convert boolean columns to binary integer
        for col in transformed_in_df.select_dtypes(include=["bool"]).columns:
            transformed_in_df[col] = transformed_in_df[col].astype("category")

        amazon_artifacts = {
            "gen_params": None,
            "tabsyn": None,
            "models": {"diffusion": None, "vae": None},
            "processing": {
                "num_col_idx": None,
                "cat_col_idx": None,
                "numpy_pandas_idx_mapping": None,
                "num_cat_processing_objects": None,
            },
        }  # saves all models artifacts as objects

        amazon_artifact_dir = chkpnt_dir or self.work_dir

        if not self.gen_name.startswith("amazon_"):
            self.logger.error(
                "The generator name does not match the Amazon Tabsyn generator."
            )
            raise ValueError(
                "The generator name does not match the Amazon Tabsyn generator."
            )
        self.logger.info(f"Started training using {self.gen_name}...")

        # POPULATE CONGURATION
        self._make_configs4amazon(
            transformed_in_df, amazon_artifact_dir
        )  # SMK function. It will update self.gen_params
        amazon_artifacts["gen_params"] = self.gen_params

        # PREPARE DATA
        (
            X_num,
            X_cat,
            numpy_pandas_idx_mapping,
            num_cat_processing_objects,
            num_col_idx,
            cat_col_idx,
        ) = self._get_numpy4amazon(transformed_in_df)
        amazon_artifacts["processing"]["num_col_idx"] = list(num_col_idx)
        amazon_artifacts["processing"]["cat_col_idx"] = list(cat_col_idx)
        amazon_artifacts["processing"][
            "numpy_pandas_idx_mapping"
        ] = numpy_pandas_idx_mapping
        amazon_artifacts["processing"][
            "num_cat_processing_objects"
        ] = num_cat_processing_objects

        # separate train and test data
        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat
        X_train_cat = X_train_cat.astype(int)
        X_test_cat = X_test_cat.astype(int)
        X_all_cat = np.vstack((X_train_cat, X_test_cat))

        def get_categories(
            X_train_cat,
        ):  # add this to the training function. It returns a list of intergers each indicating teh number of classes in the categorical column
            return (
                None
                if X_train_cat is None
                else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]
            )

        categories = get_categories(
            X_all_cat
        )  # add this to amazon_artifacts in  training function. Find the get_categories in the preprocess function ~/projects/pysdg/src/pysdg/synth/tabsyn/src/data.py
        d_numerical = X_train_num.shape[
            1
        ]  # add this scalar to amazon_artifacts in  training function

        amazon_artifacts["processing"][
            "d_numerical"
        ] = d_numerical  # TO DO: add this to the training function
        amazon_artifacts["processing"][
            "categories"
        ] = categories  # TO DO: add this to the training function

        device = torch.device(pysdg_device_name)  # SMK: May be unnecessary

        # convert to float tensor and move to CPU
        X_train_num, X_test_num = (
            torch.tensor(X_train_num).float().cpu(),
            torch.tensor(X_test_num).float().cpu(),
        )
        X_train_cat, X_test_cat = (
            torch.tensor(X_train_cat).cpu(),
            torch.tensor(X_test_cat).cpu(),
        )

        # create dataset module
        train_data = TabularDataset(X_train_num.float(), X_train_cat)

        # create train dataloader
        train_loader = DataLoader(
            train_data,
            batch_size=self.gen_params["train"]["vae"]["batch_size"],
            shuffle=True,
            num_workers=self.gen_params["train"]["vae"]["num_dataset_workers"],
        )

        # INSTANTIATE MODEL
        d_numerical = X_train_num.shape[1]
        categories = (
            None
            if X_train_cat is None
            else [len(set(X_all_cat[:, i])) for i in range(X_train_cat.shape[1])]
        )  # Supposed to be obtained from population dataset but for our purpose we obtin from train+test

        tabsyn = TabSyn(
            train_loader,
            X_test_num,
            X_test_cat,
            num_numerical_features=d_numerical,
            num_classes=categories,
            device=device,
        )

        models_dir = os.path.join(amazon_artifact_dir, "models")

        # TRAIN VAE
        # instantiate VAE model for training
        model_vae_dir = os.path.join(models_dir, "vae")
        if chkpnt_dir is None:
            tabsyn.instantiate_vae(
                **self.gen_params["model_params"],
                optim_params=self.gen_params["train"]["optim"]["vae"],
            )
            os.makedirs(model_vae_dir, exist_ok=True)

        else:
            tabsyn.instantiate_vae(
                **self.gen_params["model_params"], optim_params=None)

        # train vae model and save it
        tabsyn.train_vae(
            **self.gen_params["loss_params"],
            num_epochs=self.gen_params["train"]["vae"]["num_epochs"],
            save_path=model_vae_dir,
        )

        # embed all inputs in the latent space
        tabsyn.save_vae_embeddings(
            X_train_num, X_train_cat, vae_ckpt_dir=model_vae_dir)

        # Read vae model objects and add them to amazon_artifacts
        vae_objects = {}
        for file_name in os.listdir(model_vae_dir):
            file_path = os.path.join(model_vae_dir, file_name)
            if file_name.endswith(".pt"):
                # Save PyTorch objects
                vae_objects[file_name] = torch.load(file_path)
            elif file_name.endswith(".npy"):
                # Save NumPy objects
                vae_objects[file_name] = np.load(file_path, allow_pickle=False)

        amazon_artifacts["models"]["vae"] = vae_objects

        # TRAIN DIFFUSION
        # load latent space embeddings
        if chkpnt_dir is None:
            train_z, _ = tabsyn.load_latent_embeddings(
                model_vae_dir
            )  # train_z dim: B x in_dim
        else:
            train_z, token_dim = tabsyn.load_latent_embeddings(model_vae_dir)

        # normalize embeddings
        mean, std = train_z.mean(0), train_z.std(0)
        latent_train_data = (train_z - mean) / 2

        # create data loader
        latent_train_loader = DataLoader(
            latent_train_data,
            batch_size=self.gen_params["train"]["diffusion"]["batch_size"],
            shuffle=True,
            num_workers=self.gen_params["train"]["diffusion"]["num_dataset_workers"],
        )

        # instantiate diffusion model for training
        model_diffusion_dir = os.path.join(models_dir, "diffusion")
        if chkpnt_dir is None:
            tabsyn.instantiate_diffusion(
                in_dim=train_z.shape[1],
                hid_dim=train_z.shape[1],
                optim_params=self.gen_params["train"]["optim"]["diffusion"],
            )
            os.makedirs(model_diffusion_dir, exist_ok=True)
        else:
            tabsyn.instantiate_diffusion(
                in_dim=train_z.shape[1], hid_dim=train_z.shape[1], optim_params=None
            )
            # load state from checkpoint
            tabsyn.load_model_state(
                ckpt_dir=model_diffusion_dir, dif_ckpt_name="model.pt"
            )  # SMK may add a code here to extact the latest model or pass teh model chkpt name as argument

        # train diffusion model and save it
        tabsyn.train_diffusion(
            latent_train_loader,
            num_epochs=self.gen_params["train"]["diffusion"]["num_epochs"],
            ckpt_path=model_diffusion_dir,
        )

        # Read diffusion model objects and add them to amazon_artifacts
        diffusion_objects = {}
        for file_name in os.listdir(model_diffusion_dir):
            file_path = os.path.join(model_diffusion_dir, file_name)
            if file_name.endswith(".pt"):
                # Save PyTorch objects
                diffusion_objects[file_name] = torch.load(file_path)
            elif file_name.endswith(".npy"):
                # Save NumPy objects
                diffusion_objects[file_name] = np.load(
                    file_path, allow_pickle=False)

        amazon_artifacts["models"]["diffusion"] = diffusion_objects

        # Savve tabsyn to amazon_artifacts and to disk
        amazon_artifacts["tabsyn"] = tabsyn
        tabsyn_obj_dir = os.path.join(models_dir, "tabsyn")
        os.makedirs(tabsyn_obj_dir, exist_ok=True)
        tabsyn_save_path = os.path.join(tabsyn_obj_dir, "tabsyn.pkl")
        with open(tabsyn_save_path, "wb") as f:
            pickle.dump(tabsyn, f)

        return amazon_artifacts

    def _gen_one_synth4amazon(self, amazon_artifacts, num_rows):

        amazon_artifact_dir = (
            self.work_dir
        )  # This directory is needed as some of the artifacts need to be temporarily saved for tabsyn fuctions to work
        models_dir = os.path.join(amazon_artifact_dir, "models")
        model_vae_dir = os.path.join(models_dir, "vae")
        model_diffusion_dir = os.path.join(models_dir, "diffusion")

        if not os.path.exists(models_dir):
            # Expand amazon_artifacts["models"]["vae"] to the directory model_vae_dir
            vae_objects = amazon_artifacts["models"]["vae"]
            os.makedirs(model_vae_dir, exist_ok=True)
            for file_name, obj in vae_objects.items():
                file_path = os.path.join(model_vae_dir, file_name)
                if file_name.endswith(".pt"):
                    # Save PyTorch objects
                    torch.save(obj, file_path)
                elif file_name.endswith(".npy"):
                    # Save NumPy objects
                    np.save(file_path, obj, allow_pickle=False)

            # Expand amazon_artifacts["models"]["diffusion"] to the directory model_diffusion_dir
            diffusion_objects = amazon_artifacts["models"]["diffusion"]
            os.makedirs(model_diffusion_dir, exist_ok=True)
            for file_name, obj in diffusion_objects.items():
                file_path = os.path.join(model_diffusion_dir, file_name)
                if file_name.endswith(".pt"):
                    # Save PyTorch objects
                    torch.save(obj, file_path)
                elif file_name.endswith(".npy"):
                    # Save NumPy objects
                    np.save(file_path, obj, allow_pickle=False)

        # load latent embeddings of input data
        tabsyn = amazon_artifacts["tabsyn"]
        train_z, token_dim = tabsyn.load_latent_embeddings(model_vae_dir)

        # sample data
        # num_samples = train_z.shape[0]
        in_dim = train_z.shape[1]
        mean_input_emb = train_z.mean(0)

        mean = mean_input_emb
        sample_dim = in_dim
        x_next = sample(
            tabsyn.dif_model.denoise_fn_D, num_rows, sample_dim, device=tabsyn.device
        )
        x_next = x_next * 2 + mean.to(tabsyn.device)
        syn_data = x_next.float().cpu().numpy()

        d_numerical = amazon_artifacts["processing"][
            "d_numerical"
        ]  # TO DO: add this to the training function
        categories = amazon_artifacts["processing"][
            "categories"
        ]  # TO DO: add this to the training function

        pre_decoder = Decoder_model(
            2, d_numerical, categories, 4, n_head=1, factor=32)
        decoder_save_path = os.path.join(model_vae_dir, "decoder.pt")
        pre_decoder.load_state_dict(torch.load(decoder_save_path))

        # info["pre_decoder"] = pre_decoder

        info = {
            # "num_col_idx":amazon_artifacts["processing"]["num_col_idx"],
            # "cat_col_idx":amazon_artifacts["processing"]["cat_col_idx"],
            "token_dim": token_dim,
            "pre_decoder": pre_decoder,
        }

        num_coder = amazon_artifacts["processing"]["num_cat_processing_objects"]["num"]
        cat_coder = amazon_artifacts["processing"]["num_cat_processing_objects"]["cat"]

        syn_num, syn_cat = pysdg_split_num_cat(
            syn_data, info, num_coder, cat_coder)

        def pysdg_recover_df(syn_num, syn_cat, original_col_names, amazon_artifacts):
            numpy_pandas_idx_mapping = amazon_artifacts["processing"][
                "numpy_pandas_idx_mapping"
            ]
            recovered_df = pd.DataFrame(columns=original_col_names)
            for i in range(syn_num.shape[1]):
                numpy_index = i  # Get the index of the numpy column
                pandas_index = numpy_pandas_idx_mapping[
                    numpy_index
                ]  # Get the corresponding pandas index
                recovered_df.iloc[:, pandas_index] = syn_num[
                    :, i
                ]  # Insert the column into the recovered DataFrame
            for i in range(syn_cat.shape[1]):
                numpy_index = (
                    i + syn_num.shape[1]
                )  # Offset by the number of numeric columns
                pandas_index = numpy_pandas_idx_mapping[
                    numpy_index
                ]  # Get the corresponding pandas index
                recovered_df.iloc[:, pandas_index] = syn_cat[
                    :, i
                ]  # Insert the column into the recovered DataFrame
            return recovered_df

        syn_df = pysdg_recover_df(
            syn_num, syn_cat, list(self.enc_real.columns), amazon_artifacts
        )  # SMK: EXIT HERE

        return syn_df

    def _gen4amazon(self, amazon_artifact, num_rows, num_synths):
        synths = []
        i = 0
        # seed = 0  # default seed
        for i in range(num_synths):
            try:
                syn_synth = self._gen_one_synth4amazon(
                    amazon_artifact, num_rows)
                if not isinstance(syn_synth, pd.DataFrame):
                    raise Exception(
                        f"Generator did not succeed in generating a synth")
                self.logger.info(
                    f"Generating synth no. {i} of size {syn_synth.shape} -- Completed!"
                )
                synths.append(syn_synth)
            except Exception as e:
                self.logger.error(
                    f"Failed to generate synth no. {i} -- ☹️ -- {e}")
                continue
        return synths

    # UTILS

    def _make_writable_and_remove(self, folder_path):
        try:
            # Change permissions for the folder and its contents
            for root, dirs, files in os.walk(folder_path):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.chmod(dir_path, stat.S_IWRITE |
                             stat.S_IREAD | stat.S_IEXEC)

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)

            # Remove the folder
            shutil.rmtree(folder_path)
            self.logger.info(
                "The directory '%s' has been removed successfully.", folder_path
            )

        except Exception as e:
            self.logger.exception("Failed to remove '%s': %s", folder_path, e)

    def _save_to_zip(self, pysdg_artifacts, gen_model, zip_path):
        # Ensure the parent directory exists
        os.makedirs(self.work_dir, exist_ok=True)

        # Create a temporary directory
        try:
            with tempfile.TemporaryDirectory(dir=self.work_dir) as temp_dir:
                # Temporary file paths
                artifact_file = os.path.join(temp_dir, "pysdg_artifacts.pkl")
                model_file = os.path.join(temp_dir, "gen_model.pth")

                # Save the artifact and model separately
                with open(artifact_file, "wb") as f:
                    pickle.dump(pysdg_artifacts, f)
                torch.save(gen_model, model_file)

                # Create a ZIP file and add both files
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    zipf.write(artifact_file, arcname="pysdg_artifacts.pkl")
                    zipf.write(model_file, arcname="gen_model.pth")
                self.logger.info("Saved ZIP file: %s", zip_path)
        except Exception as e:
            self.logger.warning("Exception occurred while saving to zip: %s", e)

    def _load_from_zip(self, zip_path):

        if not zip_path.endswith(".zip") or not os.path.isfile(zip_path):
            self.logger.error(
                "The provided zip_path must be an existing .zip file.")
            raise ValueError(
                "The provided zip_path must be an existing .zip file.")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy(zip_path, os.path.join(
                temp_dir, os.path.basename(zip_path)))
            artifact_file = os.path.join(temp_dir, "pysdg_artifacts.pkl")
            model_file = os.path.join(temp_dir, "gen_model.pth")

            # Extract files from the ZIP into the temporary directory
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extract("pysdg_artifacts.pkl", path=temp_dir)
                zipf.extract("gen_model.pth", path=temp_dir)

            # Load the artifact
            with open(artifact_file, "rb") as f:
                pysdg_artifacts = pickle.load(f)

            # Load the PyTorch model
            gen_model = torch.load(model_file)

            # Return the loaded objects
            return pysdg_artifacts, gen_model

    def _purge_work_dir(self):
        try:
            for root, dirs, files in os.walk(self.work_dir):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.chmod(dir_path, stat.S_IWRITE |
                             stat.S_IREAD | stat.S_IEXEC)

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)

            shutil.rmtree(self.work_dir)
            self.logger.info(
                "The directory '%s' has been removed successfully.", self.work_dir
            )
        except Exception as e:
            self.logger.error("Failed to remove '%s': %s", self.work_dir, e)

    def _merge_dicts(self, default_params, custom_params):
        if not isinstance(custom_params, dict):
            self.logger.error(
                "User-defined hyperparmas must be provided as a dictionary."
            )
            raise TypeError(
                "User-defined hyperparmas must be provided as a dictionary."
            )

        if not custom_params:
            return default_params

        used_params = default_params.copy()

        def update_dict(d1, d2):
            """Recursively update d1 with values from d2."""
            for key, value in d2.items():
                if isinstance(value, dict):
                    update_dict(d1[key], value)
                else:
                    d1[key] = value

        update_dict(used_params, custom_params)

        def has_same_schema(d1, d2):
            """Recursively check if d2 has the same schema as d1."""
            if set(d1.keys()) != set(d2.keys()):
                return False
            for key in d1:
                if isinstance(d1[key], dict):
                    if not isinstance(d2[key], dict) or not has_same_schema(
                        d1[key], d2[key]
                    ):
                        return False
            return True

        if not has_same_schema(default_params, used_params):
            self.logger.error(
                "The user-defined parameter dictioanry does not match the correct schema."
            )
            raise ValueError(
                "The user-defined parameter dictioanry does not match the correct schema."
            )

        return used_params
