gpu=0
dataset=profession
model=llama31-Instruct-4bit        # LLM model
K=50                               # S the number of selected words used by the LLM to synthesize texts
gen_len=512                        # L the length of texts synthesized by the LLM
part_of_dataset=0.33               # Synthesized quantity: the ratio of the number of synthesized texts to that of the original dataset
aaai_num=4                         # best BTM iterate times
num_keywords=60                    # K the number of keywords for each utterance
niter=50                           # T the number of iterations for the Gibbs sampling process

echo "Using ${dataset} data set..."
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run.sh ${dataset} ${num_keywords} ${niter}

start=$(date +%s)
CUDA_VISIBLE_DEVICES=${gpu} python PEARL/keywords_sents_generate.py --dataset_name ${dataset} --model_name ${model} --K ${K} --aaai_num ${aaai_num} --gen_len ${gen_len} --gen_len ${gen_len} --part_of_dataset ${part_of_dataset} --num_keywords ${num_keywords} --niter ${niter}
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run_add.sh ${dataset} "${model}_1_${niter}_${num_keywords}_${K}_${gen_len}_${part_of_dataset}"  ${num_keywords} ${niter}
end=$(date +%s)
runtime=$((end-start))
echo "Execution Time iteration 1: $runtime seconds"

start=$(date +%s)
CUDA_VISIBLE_DEVICES=${gpu} python PEARL/keywords_sents_generate.py --dataset_name ${dataset} --model_name ${model} --K ${K} --aaai_num ${aaai_num} --iter 1 --gen_len ${gen_len} --gen_len ${gen_len} --part_of_dataset ${part_of_dataset} --num_keywords ${num_keywords} --niter ${niter}
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run_add.sh ${dataset} "${model}_2_${niter}_${num_keywords}_${K}_${gen_len}_${part_of_dataset}"  ${num_keywords} ${niter}
end=$(date +%s)
runtime=$((end-start))
echo "Execution Time iteration 2: $runtime seconds"

start=$(date +%s)
CUDA_VISIBLE_DEVICES=${gpu} python PEARL/keywords_sents_generate.py --dataset_name ${dataset} --model_name ${model} --K ${K} --aaai_num ${aaai_num} --iter 2 --gen_len ${gen_len} --gen_len ${gen_len} --part_of_dataset ${part_of_dataset} --num_keywords ${num_keywords} --niter ${niter}
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run_add.sh ${dataset} "${model}_3_${niter}_${num_keywords}_${K}_${gen_len}_${part_of_dataset}"  ${num_keywords} ${niter}
end=$(date +%s)
runtime=$((end-start))
echo "Execution Time iteration 3: $runtime seconds"

start=$(date +%s)
CUDA_VISIBLE_DEVICES=${gpu} python PEARL/keywords_sents_generate.py --dataset_name ${dataset} --model_name ${model} --K ${K} --aaai_num ${aaai_num} --iter 3 --gen_len ${gen_len} --gen_len ${gen_len} --part_of_dataset ${part_of_dataset} --num_keywords ${num_keywords} --niter ${niter}
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run_add.sh ${dataset} "${model}_4_${niter}_${num_keywords}_${K}_${gen_len}_${part_of_dataset}"  ${num_keywords} ${niter}
end=$(date +%s)
runtime=$((end-start))
echo "Execution Time iteration 4: $runtime seconds"

CUDA_VISIBLE_DEVICES=${gpu} python PEARL/keywords_sents_generate.py --dataset_name ${dataset} --model_name ${model} --K ${K} --aaai_num ${aaai_num} --iter 4 --gen_len ${gen_len} --gen_len ${gen_len} --part_of_dataset ${part_of_dataset} --num_keywords ${num_keywords} --niter ${niter}
CUDA_VISIBLE_DEVICES=${gpu} bash PEARL/script/run_add.sh ${dataset} "${model}_5_${niter}_${num_keywords}_${K}_${gen_len}_${part_of_dataset}"  ${num_keywords} ${niter}
end=$(date +%s)
runtime=$((end-start))
echo "Execution Time iteration 5: $runtime seconds"
