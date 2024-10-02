import subprocess


def get_commit_hash():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.decode("utf-8").replace("\n", "")


def is_clean_working_tree():
    return len(get_working_tree_diff()) == 0


def get_working_tree_diff():
    return subprocess.run(["git", "diff", "--no-color"], capture_output=True).stdout.decode("utf-8")[:-1]


def merge_update(base: dict, update: dict):
    output = dict()
    keys = set(base.keys()) | set(update.keys())
    for k in keys:
        v = base.get(k, None)
        update_v = update.get(k, None)
        if isinstance(v, dict):
            assert update_v is None or isinstance(update_v, dict)
            update_v = update_v or dict()
            output[k] = merge_update(v, update_v)
        else:
            output[k] = update_v if update_v is not None else v
    return output