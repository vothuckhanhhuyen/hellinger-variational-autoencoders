#!/usr/bin/env bash
set -eo pipefail
shopt -s nullglob globstar

# define TMPDIR, if it's empty
if [[ -z "$TMPDIR" ]]; then
    TMPDIR="/tmp"
fi
echo "TMPDIR: $TMPDIR"

# activate conda env
eval "$(conda shell.bash hook)"
conda activate mopoe
echo "CONDA_PREFIX: $CONDA_PREFIX"

TEMP_DIR="" # change to your own temp dir for data storage
METHOD="helvae"  # NOTE: valid options are "joint_elbo", "poe", "moe", "jsd", "helvae", "joint_helvae"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="categorical"
DIR_DATA="$TEMP_DIR/data/CelebA"
DIR_TEXT="$TEMP_DIR/data/CelebA"
DIR_CLF="$TEMP_DIR/trained_classifiers/trained_clfs_celeba"
PATH_INC_V3="$TEMP_DIR/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TMPDIR/CelebA"

for seed in 0 1 2; do
for beta in 1 2.5 5; do
DIR_EXPERIMENT="$PWD/runs/CelebA/$METHOD/factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}/seed_${seed}/beta_${beta}"
python main_celeba.py --dir_data=$TEMP_DIR/data \
	              --dir_text="$DIR_TEXT" \
	              --dir_clf="$DIR_CLF" \
	              --dir_experiment="$DIR_EXPERIMENT" \
	              --inception_state_dict="$PATH_INC_V3" \
	              --dir_fid=$DIR_FID \
                  --method=$METHOD \
	              --beta=$beta \
	              --beta_style=2.0 \
	              --beta_content=1.0 \
	              --beta_m1_style=1.0 \
	              --beta_m2_style=5.0 \
	              --div_weight_m1_content=0.35 \
	              --div_weight_m2_content=0.35 \
	              --div_weight_uniform_content=0.3 \
	              --likelihood_m1=$LIKELIHOOD_M1 \
	              --likelihood_m2=$LIKELIHOOD_M2 \
	              --batch_size=256 \
	              --initial_learning_rate=0.00005 \
	              --eval_freq=50 \
	              --eval_freq_fid=50 \
	              --end_epoch=200 \
	              --factorized_representation \
	              --calc_nll \
	              --eval_lr \
				  --use_clf \
	              --calc_prd \
			      --ver_type="f64" \
				  --seed=$seed
done
done
