# https://github.com/tricktreat/piqn/tree/main
# http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz
import os
import pathlib

try:
    import gdown
except ImportError:
    print(f"gdown must be installed: 'pip install gdown'")
    exit(1)

assert (pathlib.Path.cwd() / ".git").exists(), \
    f"Script must be run from the project root directory, we are in {os.getcwd()}"

genia_folder_name = "1krNw98zi5mp0KPZGoCo5D5ne8dWV6pUD"
genia_local_folder_path = "datasets/genia/."
gdown.download_folder(id=genia_folder_name, output=genia_local_folder_path, quiet=False, use_cookies=False, verify=True)
print(f"Downloaded GENIA corpus to {genia_local_folder_path}")
