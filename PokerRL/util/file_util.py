# Copyright (c) 2019 Eric Steinberger


"""
Some os-agnostic utility functions to read and write files easily.
"""
import json
import os
import pickle
from os.path import join as ospj


def create_dir_if_not_exist(path):
    if (not os.path.exists(path)) and (not os.path.isfile(path)):
        os.makedirs(path)


def get_all_files_in_dir(_dir):
    return [f for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f))]


def get_all_dirs_in_dir(_dir):
    return [d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]


def get_file_name_without_ending_and_path_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def write_dict_to_file_json(_dir, file_name, dictionary):
    create_dir_if_not_exist(_dir)
    with open(ospj(_dir, str(file_name) + ".json"), 'w') as file:
        file.write(json.dumps(dictionary))


def write_dict_to_file_js(_dir, file_name, dictionary):
    create_dir_if_not_exist(_dir)
    with open(ospj(_dir, str(file_name) + ".js"), 'w') as file:
        file.write("const data=" + json.dumps(dictionary))


def do_pickle(obj, path, file_name):
    create_dir_if_not_exist(path)
    with open(ospj(path, str(file_name) + ".pkl"), "wb") as pkl_file:
        pickle.dump(obj=obj, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, file_name=None):
    if file_name is None:
        p = path
    else:
        p = ospj(path, str(file_name) + ".pkl")

    with open(p, "rb") as pkl_file:
        state = pickle.load(pkl_file)
    return state
