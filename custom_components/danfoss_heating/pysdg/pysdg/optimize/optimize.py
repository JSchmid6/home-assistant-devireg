import optuna
import logging
import sys
import pandas as pd

from typing import Callable
from pysdg.gen.generate import Generator
from functools import partial
from optuna.trial import Trial
from copy import deepcopy


class BayesianOptimizationRoutine:
    """A class to perform Bayesian optimization for various synthetic data generators

    Attributes:
        PARAMETER_RANGES (dict): A dictionary containing parameter ranges for different generators.
        gen (Generator): The generator object to be optimized.
        eval_function (Callable): The evaluation function to assess generator performance.
        holdout_df (pd.DataFrame | None): A DataFrame for holdout validation, if any.
        objective (str): The optimization objective, either "maximize" or "minimize".
        n_trials (int): The number of optimization trials to run.
        study_name (str): The name of the Optuna study.
        dump_sqlite (bool): Whether to dump the study results to an SQLite database.
        dump_csv (bool): Whether to dump the study results to a CSV file.
        best_gen (Generator | None): The best generator found during optimization.
        gen_name (str): The name of the generator being optimized.
        study (optuna.study.Study): The Optuna study object.
        black_box_function (Callable): The black-box function for optimization.
    """


    PARAMETER_RANGES = {

        "synthcity_ctgan": {

            "generator_n_layers_hidden": ("int", 1, 6),
            "generator_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "generator_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "generator_dropout": ("float", 0.05, 0.3),

            "discriminator_n_layers_hidden": ("int", 1, 6),
            "discriminator_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "discriminator_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "discriminator_dropout": ("float", 0.05, 0.3),

            "n_iter": ("categorical", [10, 15, 25, 50]),
            "lr": ("categorical", [1e-5, 1e-4, 1e-3, 1e-2]),
            "weight_decay": ("float", 1e-5, 1e-2),
            "batch_size": ("categorical", [32, 64, 128, 256, 512]),
            "clipping_value": ("categorical", [-1, 0, 1]),
            "encoder_max_clusters": ("int", 5, 50),
            "adjust_inference_sampling": ("categorical", [True, False]),
        },

        "synthcity_bayesian_network": {
            "struct_learning_search_method": ("categorical", ["hillclimb", "pc", "tree_search", "mmhc", "exhaustive"]),
            "struct_learning_score": ("categorical", ["bdeu", "k2", "bds", "bic"]),
            "struct_max_indegree": ("int", 1, 10),
            "encoder_max_clusters": ("int", 1, 50),
            "encoder_noise_scale": ("float", 0.01, 0.25),
        },

        "synthcity_tvae": {

            "decoder_n_layers_hidden": ("int", 1, 6),
            "decoder_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "decoder_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "decoder_dropout": ("float", 0.05, 0.3),

            "encoder_n_layers_hidden": ("int", 1, 6),
            "encoder_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "encoder_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "encoder_dropout": ("float", 0.05, 0.3),

            "n_iter": ("categorical", [10, 15, 25, 50]),
            "lr": ("categorical", [1e-5, 1e-4, 1e-3, 1e-2]),
            "weight_decay": ("float", 1e-5, 1e-2),
            "batch_size": ("categorical", [32, 64, 128, 256, 512]),
            "n_iter_min": ("int", 5, 5),
            "patience": ("int", 5, 5),
        },

        "synthcity_rtvae": {

            "decoder_n_layers_hidden": ("int", 1, 6),
            "decoder_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "decoder_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "decoder_dropout": ("float", 0.05, 0.3),

            "encoder_n_layers_hidden": ("int", 1, 6),
            "encoder_n_units_hidden": ("categorical", [32, 64, 128, 256]),
            "encoder_nonlin": ("categorical", ["elu", "relu", "selu", "leaky_relu"]),
            "encoder_dropout": ("float", 0.05, 0.3),

            "n_iter": ("categorical", [10, 15, 25, 50]),
            "lr": ("categorical", [1e-5, 1e-4, 1e-3, 1e-2]),
            "weight_decay": ("float", 1e-5, 1e-2),
            "batch_size": ("categorical", [32, 64, 128, 256, 512]),
            "n_iter_min": ("int", 5, 5),
            "patience": ("int", 5, 5),
        },

        "synthcity_arf": {
            "num_trees": ("int", 10, 250),
            "delta": ("int", 0, 0),
            "max_iters": ("int", 10, 100),
            "early_stop": ("categorical", [True, False]),
            "min_node_size": ("int", 2, 10),
            "sampling_patience": ("int", 100, 1000),
        },
        "synthcity_nflow": {
             "n_iter": ("categorical", [10, 15, 25, 50]),
             "n_layers_hidden": ("int", 1, 6),
             "n_units_hidden": ("categorical", [64, 128, 256]),
             "batch_size": ("categorical", [32, 64, 128, 256, 512]),
             "num_transform_blocks": ("categorical", [1, 2, 5, 10, 25, 50]),
             "dropout": ("float", 0.05, 0.3),
             "batch_norm": ("categorical", [True, False]),
             "num_bins": ("categorical", [1, 2, 5, 10, 25, 50]),
             "tail_bound": ("int", 2, 10),
             "lr": ("categorical", [1e-5, 1e-4, 1e-3, 1e-2]),
             "apply_unconditional_transform": ("categorical", [True, False]),
             "linear_transform_type": ("categorical", ["lu", "permutation", "svd"]),
             "base_transform_type": ("categorical", [
                 "rq-coupling",
                 "affine-autoregressive",
                 "quadratic-autoregressive",
                 "rq-autoregressive"
             ]),
             "n_iter_min": ("int", 5, 5),
             "patience": ("int", 5, 5),
         }
    }
    def __init__(
        self,
        gen: Generator,
        eval_function: Callable,
        holdout_df: pd.DataFrame | None = None,
        objective: str = "maximize",
        n_trials: int = 10,
        study_name: str = "my_study",
        dump_sqlite: bool = False,
        dump_csv: bool = False):
        """Intialize BayesianOptimizationRoutine.

        Args:
            gen (Generator): The generator object to be optimized.
            eval_function (Callable): The evaluation function to assess generator performance.
            holdout_df (pd.DataFrame | None, optional): A DataFrame for holdout validation, if any. Defaults to None.
            objective (str, optional): The optimization objective, either "maximize" or "minimize". Defaults to "maximize".
            n_trials (int, optional): The number of optimization trials to run. Defaults to 10.
            study_name (str, optional): The name of the Optuna study. Defaults to "my_study".
            dump_sqlite (bool, optional): Whether to dump the study results to an SQLite database. Defaults to False.
            dump_csv (bool, optional): Whether to dump the study results to a CSV file. Defaults to False.

        """

        """
        Run the Bayesian optimization routine.
        """
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.gen = gen
        self.eval_function = eval_function
        self.holdout_df = holdout_df
        self.objective = objective
        self.n_trials = n_trials
        self.study_name = study_name
        self.dump_sqlite = dump_sqlite
        self.dump_csv = dump_csv
        self.best_gen = None

        self.gen_name: str = gen.gen_name
        if self.gen_name not in self.PARAMETER_RANGES:
            raise ValueError(f"Unsupported generator {self.gen_name}")

        # create the study
        storage = f"sqlite:///{self.study_name}.db" if dump_sqlite else None
        self.study = optuna.create_study(direction=self.objective, storage=storage)

        # run the optimization routine
        self.black_box_function = partial(self.generic_black_box_function, gen=self.gen)
        self.study.optimize(self.black_box_function, n_trials=self.n_trials)
        # retrain the best generator
        self.retrain_generator(self.study.best_params)

    def generate_params(self, trial: Trial, generator_name: str):
        """
        Generate parameters for the generator using the trial object.
        """
        paramater_dict = self.PARAMETER_RANGES[generator_name]
        params = {}

        for param, (ptype, *args) in paramater_dict.items():
            if ptype == "int":
                params[param] = trial.suggest_int(param, *args)
            elif ptype == "categorical":
                params[param] = trial.suggest_categorical(param, *args)
            elif ptype == "float":
                params[param] = trial.suggest_float(param, *args)
            else:
                raise ValueError(f"Unknown parameter type: {ptype}")

        return params

    def generic_black_box_function(self, trial: Trial, gen: Generator):
        """
        Run one training iteration of the generator and evaluate the performance.
        """
        params = self.generate_params(trial, gen.gen_name)

        gen.gen_params = params
        gen.train()
        gen.gen(num_rows=len(gen.enc_real), num_synths=1)

        if isinstance(self.holdout_df, pd.DataFrame) and not self.holdout_df.empty:
            metric = self.eval_function(gen, self.holdout_df)
        else:
            metric = self.eval_function(gen)

        trial.set_user_attr(self.eval_function.__name__, metric)
        return metric

    def get_optimization_results(self):
        """Returns the optimization results as a DataFrame."""
        return self.study.trials_dataframe()

    def retrain_generator(self, params):
        """Retrain the best generator using the best parameters."""
        self.best_gen = deepcopy(self.gen)
        self.best_gen.gen_params = params
        self.best_gen.train()