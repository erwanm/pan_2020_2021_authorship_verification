
DIR="/home/moreaue/clg-authorship-experiments/pan_2020_2021_authorship_verification"


if [ $# -ne 1 ]; then
    echo "arg: <input prefix>" 1>&2
    exit 1
fi
prefix=$1

if [ ! -f "$prefix.train.jsonl" ] ||  [ ! -f "$prefix.test.jsonl" ]; then
    echo " File $prefix.train.jsonl or $prefix.test.jsonl missing." 1>&2
    exit 1
fi
if [ ! -f "$prefix.train-truth.jsonl" ] ||  [ ! -f "$prefix.test-truth.jsonl" ]; then
    echo " File $prefix.train-truth.jsonl or $prefix.test-truth.jsonl missing." 1>&2
    exit 1
fi

tmpdir=$(mktemp --tmpdir=/tmp -d "process.XXXXXXXXXX")
mkdir $tmpdir
echo "tmpdir=$tmpdir" 1>&2
cd $tmpdir
cp -R "$DIR/helper_functions" .
cp -R "$DIR/preprocessing" .
cp -R "$DIR/training_adhominem" .
cp -R "$DIR/training_o2d2" .
cp -R "$DIR/inference" .

mkdir "data_original"
cd "data_original"
ln -s "$DIR/data_original/cc.en.300.bin"
cd ../preprocessing


# preprocessing

echo "step 1" 1>&2
python step1-specified-train-test.py "$prefix.train" "$prefix.test"
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
echo "step 2" 1>&2
python step2_preprocess.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
echo "step 3" 1>&2
python step3_count.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
echo "step 4" 1>&2
python step4_make_vocabularies.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
echo "steps 5-7" 1>&2
python step5_sample_pairs_cal.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
python step6-exact-test-set.py "$prefix.test"
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
python step7_sample_pairs_dev.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi
echo "7 steps done, copying preprocessing dir" 1>&2
rm -rf "$prefix.data_preprocessed"
cp -R "$tmpdir/data_preprocessed" "$prefix.data_preprocessed"

echo "training model, part 1..." 1>&2

cd $tmpdir/training_adhominem
python train_adhominem.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi

echo "copying model 1..." 1>&2
rm -rf "$prefix.results_adhominem"
cp -R "$tmpdir/results_adhominem" "$prefix.results_adhominem"


echo "training model, part 2..." 1>&2

cd $tmpdir/training_o2d2
python train_o2d2.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi

echo "copying model 2..." 1>&2
rm -rf "$prefix.results_o2d2"
cp -R "$tmpdir/results_o2d2" "$prefix.results_o2d2"


echo "inference regular..." 1>&2
cd $tmpdir/inference
python run_inference.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi

echo "getting predicted regular.." 1>&2
cat predicted.tsv >"$prefix.predicted.regular.tsv"
rm -f predicted.tsv

echo "inference special ..."

rm -rf "../results_adhominem" "../results_o2d2"
cp -R "$DIR/pretrained_models/results_adhominem" "$DIR/pretrained_models/results_o2d2" ..
python run_inference.py
if [ $? -ne 0 ]; then
    echo "ERROR" 1>&2
    exit 1
fi

echo "getting predicted special.." 1>&2
cat predicted.tsv >"$prefix.predicted.special.tsv"


rm -rf $tmpdir
echo "done." 1>&2


