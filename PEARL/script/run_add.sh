#!/bin/bash
start=$(date +%s)

dataset_name=${1}
model_info=${2}
num_keywords=${3} # how many keywords in one doc
niter=${4}  # Gibbs iterate times
python ./PEARL/pyy/pearl.py --dataset-name $dataset_name --datasets-dir "./data/" --data-form ${model_info} --num-keywords ${num_keywords} --niter ${niter}
classes_pt=./data/${dataset_name}/classes.txt
K=`wc -l < $classes_pt`   # number of topics
alpha=`echo "scale=3;50/$K"|bc`     # bc float cacluate
beta=0.01
E=8  # BTM iterate times


data_dir=./data/${dataset_name}/info_${model_info}/
model_dir=${data_dir}model/
mkdir -p ${data_dir}/model
dwid_pt=${data_dir}doc_wids.txt   # docs with id
cos_sim_pt=${data_dir}cos_sim.txt
biterm_num=`wc -l < ${cos_sim_pt}`
voca_pt=${data_dir}voca.txt   # vocabulary
W=`wc -l < $voca_pt` # vocabulary size

make -C ./PEARL/cppp
./PEARL/cppp/btm sum_b $K $W $alpha $beta $E $niter $dwid_pt $model_dir $cos_sim_pt $biterm_num

end=$(date +%s)
runtime=$((end-start))
echo "Execution Time: $runtime seconds"

python ./PEARL/pyy/evaluation.py --dataset $dataset_name --class-num $K --E $E --data-form ${model_info} --num_keywords ${num_keywords} --niter ${niter}