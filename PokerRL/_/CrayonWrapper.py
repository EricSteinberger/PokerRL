# Copyright (c) 2019 Eric Steinberger


from os.path import join as ospj

from pycrayon import CrayonClient

from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.util.file_util import create_dir_if_not_exist, write_dict_to_file_json


class CrayonWrapper:
    """
    Wraps PyCrayon (https://github.com/torrvision/crayon), a language-agnostic interface to TensorBoard.
    """

    def __init__(self, name, runs_distributed, runs_cluster, chief_handle, path_log_storage=None,
                 crayon_server_address="localhost"):
        self._name = name
        self._path_log_storage = path_log_storage
        if path_log_storage is not None:
            create_dir_if_not_exist(path_log_storage)

        self._chief_handle = chief_handle
        self._crayon = CrayonClient(hostname=crayon_server_address)
        self._experiments = {}
        self.clear()
        self._custom_logs = {}  # dict of exps containing dict of graph names containing lists of {step: val, } dicts

        self._ray = MaybeRay(runs_distributed=runs_distributed, runs_cluster=runs_cluster)

    @property
    def name(self):
        return self._name

    @property
    def path_log_storage(self):
        return self._path_log_storage

    def clear(self):
        """
        Does NOT clear crayon's internal experiment logs and files.
        """
        self._experiments = {}

    def export_all(self, iter_nr):
        """
        Exports all logs of the current run in Tensorboard's format and as json files.
        """
        if self._path_log_storage is not None:
            path_crayon = ospj(self._path_log_storage, str(self._name), str(iter_nr), "crayon")
            path_json = ospj(self._path_log_storage, str(self._name), str(iter_nr), "as_json")
            create_dir_if_not_exist(path=path_crayon)
            create_dir_if_not_exist(path=path_json)
            for e in self._experiments.values():
                e.to_zip(filename=ospj(path_crayon, e.xp_name + ".zip"))
                write_dict_to_file_json(dictionary=self._custom_logs, _dir=path_json, file_name="logs")

    def update_from_log_buffer(self):
        """
        Pulls newly added logs from the chief onto whatever worker CrayonWrapper runs on. It then adds all these new
        logs to Tensorboard (i.e. PyCrayon's docker container)
        """
        new_v, exp_names = self._get_new_vals()

        for e in exp_names:
            if e not in self._experiments.keys():
                self._custom_logs[e] = {}
                try:
                    self._experiments[e] = self._crayon.create_experiment(xp_name=e)
                except ValueError:
                    self._crayon.remove_experiment(xp_name=e)
                    self._experiments[e] = self._crayon.create_experiment(xp_name=e)

        for name, vals_dict in new_v.items():
            for graph_name, data_points in vals_dict.items():

                for data_point in data_points:
                    step = int(data_point[0])
                    val = data_point[1]

                    self._experiments[name].add_scalar_value(name=graph_name, step=step, value=val)
                    if graph_name not in self._custom_logs[name].keys():
                        self._custom_logs[name][graph_name] = []

                    self._custom_logs[name][graph_name].append({step: val})

    def _get_new_vals(self):
        """
        Returns:
            dict: Pulls and returns newly added logs from the chief onto whatever worker CrayonWrapper runs on.
        """
        return self._ray.get(self._ray.remote(self._chief_handle.get_new_values))
