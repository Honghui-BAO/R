#!/bin/bash
array=(0 1 2 3 4 5 6 7)
nn=8

for category in "CDs_and_Vinyl"; do
    path_to_data=data
    train_file=$(ls -f ${path_to_data}/train/${category}*11.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*11.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*11.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    base_model=output_dir/${category}/latent
    for lr in 1e-4; do
        # model_ckpt=output_dir/CDs_and_Vinyl/both_beta0.01_lr1e-5_gen8_iter1
        model_ckpt=output_dir/CDs_and_Vinyl/nomul_attn_lr_1e-5/checkpoint-1400
        # model_ckpt=/storage_fast/hhbao/project/R3/output_dir/CDs_and_Vinyl/latent/checkpoint-770
        mkdir -p ./output_dir/${category}_lr_test_base
        python src/utils/split.py --input_path $test_file \
                --output_path ./output_dir/${category}_lr_test_base --nn ${nn}

        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo output_dir/${category}_lr_test_base/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 4 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ./output_dir/${category}_lr_test_base/${i}.csv \
                    --result_json_data ./output_dir/${category}_lr_test_base/${i}.json &
        done
        wait
        python src/utils/merge.py --input_path ./output_dir/${category}_lr_test_base/ \
                --output_path ${model_ckpt}/final_result.json --nn ${nn}
                
        python src/utils/calc.py --path ${model_ckpt}/final_result.json \
                --item_path ./data/info/CDs_and_Vinyl_5_2015-10-2018-11.txt
    done

done
