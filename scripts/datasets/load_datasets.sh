test -d ".git" || { echo "This script must be executed from the project root directory"; exit 1; }
set -xe
test -d "datasets" || mkdir -v "datasets"

WORK_DIR=$(pwd)
DATASET_DIR="$WORK_DIR/datasets"

# conll03
function load_conll03 () {
  mkdir "$DATASET_DIR/conll03"
  test -f "$DATASET_DIR/conll03.zip" || wget https://polybox.ethz.ch/index.php/s/bFf8vJBonIT7sr8/download -O "$DATASET_DIR/conll03.zip"
  unzip -oq "$DATASET_DIR/conll03.zip" -d "$DATASET_DIR"
  mv "$DATASET_DIR/conll03_ner/"* "$DATASET_DIR/conll03"
  rm -rv "$DATASET_DIR/conll03_ner"
  python3 "$WORK_DIR/scripts/datasets/load_conll03_doclvl.py"
  rm -fv $DATASET_DIR/conll03/*.txt
  rm -v "$DATASET_DIR/conll03.zip"
}

# conll04
function load_conll04 () {
  mkdir -v "$DATASET_DIR/conll04"
  wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/ -P "$DATASET_DIR/conll04"
}

# ade
function load_ade () {
  mkdir -v "$DATASET_DIR/ade"
  wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ -P "$DATASET_DIR/ade"
  python3 "$WORK_DIR/scripts/datasets/load_ade.py"
}

# genia
function load_genia () {
  mkdir -v "$DATASET_DIR/genia"
  python3 "$WORK_DIR/scripts/datasets/load_genia.py"
}

# nyt
function load_nyt () {
  types=( "dev" "test" "train" )
  mkdir -v "$DATASET_DIR/nyt"
  for type in "${types[@]}"; do
    wget "https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/$type.json" -P "$DATASET_DIR/nyt"
  done
  python3 "$WORK_DIR/scripts/datasets/load_nyt.py"
}

# scierc
function load_scierc () {
  mkdir -v "$DATASET_DIR/scierc"
  wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/ -P "$DATASET_DIR/scierc"
}

# ace05
# following instructions from https://github.com/luanyi/DyGIE/tree/master/preprocessing
function preprocess_ace05 () {
  test -d "$DATASET_DIR/ace05/dygie" || git clone https://github.com/luanyi/DyGIE.git datasets/ace05/dygie
  test -f "$DATASET_DIR/ace05/dygie/preprocessing/common/manual_fix.py" || wget https://raw.githubusercontent.com/btaille/sincere/dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31/ace_preprocessing/ace2005/manual_fix.py -O "$DATASET_DIR/ace05/dygie/preprocessing/common/manual_fix.py"
  test -f "$DATASET_DIR/ace05/ace05_to_json.py" || wget https://raw.githubusercontent.com/lyutyuh/ASP/master/data/ace05_ere/ace05_to_json.py -O "$DATASET_DIR/ace05/ace05_to_json.py"
  sed -i 's/dir_path = os.path.dirname(os.path.realpath(__file__))/import sys; dir_path = sys.argv[1]/' "$DATASET_DIR/ace05/ace05_to_json.py"
  cp -r "$DATASET_DIR"/ace05/*/English/ "$DATASET_DIR/ace05/dygie/preprocessing/ace2005"
  pushd "$DATASET_DIR/ace05/dygie/preprocessing/common"
  test -d "stanford-corenlp-full-2015-04-20" || wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
  test -d "stanford-postagger-2015-04-20" || wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
  test -d "stanford-corenlp-full-2015-04-20" || unzip stanford-corenlp-full-2015-04-20.zip
  test -d "stanford-postagger-2015-04-20" || unzip stanford-postagger-2015-04-20.zip
  popd
  pushd "$DATASET_DIR/ace05/dygie/preprocessing/ace2005"
  for f in English/*.zip; do test -d "English/$(basename ${f%.zip})" || unzip "$f" -d "English/$(basename ${f%.zip})" || echo ""; done
  bash "$WORK_DIR/scripts/datasets/load_ace05.sh"
  sed -i 's/print fn/print(fn)/' "$DATASET_DIR/ace05/dygie/preprocessing/ace2005/ace2json.py"
  mkdir -p "../../data/ace05/json"
  python3 ace2json.py
  popd
  python3 "$DATASET_DIR/ace05/ace05_to_json.py" "$DATASET_DIR/ace05/dygie/data/ace05/json/"
  mv "$DATASET_DIR/ace05/dygie/data/ace05/json/ace05_*.json" "$DATASET_DIR/ace05"
  python3 "$WORK_DIR/scripts/datasets/load_ace05.py" "--path" "$DATASET_DIR/ace05" "--update_truecase"
}

function load_ace05 () {
  mkdir -vp "$DATASET_DIR/ace05"
  test -f "$DATASET_DIR/ace05/ace05_types.json" || preprocess_ace05
  mv $DATASET_DIR/ace05/dygie/data/ace05/json/ace05_*.json "$DATASET_DIR/ace05"
}

test -d "datasets/conll03" || load_conll03
test -d "datasets/conll04" || load_conll04
test -d "datasets/ade" || load_ade
test -d "datasets/genia" || load_genia
test -d "datasets/nyt" || load_nyt
test -d "datasets/scierc" || load_scierc
test -f "datasets/ace05/ace05_types.json" || load_ace05
