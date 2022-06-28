
DIR="/home/moreaue/clg-authorship-experiments/pan_2020_2021_authorship_verification"

if [ $# -ne 3 ]; then
    echo "arg: <ABSOLUTE PATH input dir> <compute|long> <mem e.g. 50G>" 1>&2
    exit 1
fi
inputdir="$1"
parti="$2"
mem="$3"

for expedir in "$inputdir"/*; do
    if [ -d "$expedir" ]; then
	for f in "$expedir"/*.train-truth.jsonl; do
	    prefix=${f%.train-truth.jsonl}
	    
	    echo "#!/bin/bash" > "$prefix.preprocess.sbatch"
	    echo "#SBATCH -p $parti" >> "$prefix.preprocess.sbatch"
	    echo "#SBATCH --gres gpu:rtx2080ti:1" >> "$prefix.preprocess.sbatch"
	    echo "#SBATCH --mem $mem" >> "$prefix.preprocess.sbatch"
	    echo "#SBATCH -J full" >> "$prefix.preprocess.sbatch"
	    echo 'eval "$(~/anaconda3/bin/conda shell.bash hook)"' >> "$prefix.preprocess.sbatch"
	    echo "conda run -n PAN21 $DIR/run_full_process.sh $prefix" >> "$prefix.preprocess.sbatch"

	done
    fi
done
