from entropic_map import EntropicMap
from exp_configs import EXP_GROUPS
from get_data import get_data
import argparse
import datetime
import os
import pickle
import shutil
import json

if __name__ == "__main__":
    # CUDA_LAUNCH_BLOCKING=1
    import exp_configs

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_name', required=True)
    parser.add_argument('-ep', '--exp_path', required=True)
    args, others = parser.parse_known_args()

    if not os.path.exists("runs"):
        os.mkdir("runs")
    exp_path = "runs/" + args.exp_path
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_name = args.exp_name

    for param_dict in EXP_GROUPS[exp_name]:
        Map = EntropicMap()
        data_dict = get_data(**param_dict["data_params"])
        X, Y = data_dict["X"], data_dict["Y"]
        if param_dict["data_params"]["use_fused"]:
            X_tilde, Y_tilde = data_dict["X_tilde"], data_dict["Y_tilde"]
        else:
            X_tilde, Y_tilde = None, None
        if param_dict["data_params"]["use_validation"]:
            X_val, Y_val, X_test, Y_test, labels = data_dict["X_val"], data_dict["Y_val"],\
                data_dict["X_test"], data_dict["Y_test"], data_dict["labels"]
        else:
            X_val, Y_val, X_test, Y_test, labels = None, None, None, None, None

        path = exp_path
        path_components = [
            param_name + "_" + str(param_dict["ent_map_params"][param_name]) for param_name in param_dict["path_params"]
        ]
        path_components = "_".join(path_components)

        path += "/" + path_components
        path += "_" + str(datetime.datetime.now())[:19]
        os.mkdir(path)

        shutil.copyfile(r"exp_configs.py", r"{}".format(path) + "/exp_configs.py")
        shutil.copyfile(r"entropic_map.py", r"{}".format(path) + "/entropic_map.py")
        shutil.copyfile(r"tools.py", r"{}".format(path) + "/tools.py")
        shutil.copyfile(r"scores.py", r"{}".format(path) + "/scores.py")
        shutil.copyfile(r"scores_fused.py", r"{}".format(path) + "/scores_fused.py")
        shutil.copyfile(r"train.py", r"{}".format(path) + "/train.py")
        shutil.copyfile(r"get_data.py", r"{}".format(path) + "/get_data.py")
        shutil.copyfile(r"gw.py", r"{}".format(path) + "/gw.py")
        shutil.copyfile(r"fgw.py", r"{}".format(path) + "/fgw.py")

        print(X)
        print(Y)
        print(param_dict)
        with open(os.path.join(path, 'param_dict.json'), 'w') as fp:
            json.dump(param_dict, fp, indent=4)

        Map.fit(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test,
                labels=labels,
                logdir=path, **param_dict["ent_map_params"],
                **param_dict["data_params"])
