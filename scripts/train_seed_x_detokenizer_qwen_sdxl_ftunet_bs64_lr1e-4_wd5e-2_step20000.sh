
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe

echo '--------------------------'
# ps -elf | grep python3.8 | grep -v grep | awk '{print $4}' | xargs kill -9

echo "start training"
# export ASCEND_LAUNCH_BLOCKING=1
PROJ_PATH='.'
WFS_ROOT='/mnt/wfs/mmshanghai8wfssh/project_mm-base-vision-tj/huangzp/'



# change: OUTPUT_PATH, save_steps, resume_from_checkpoint
exp_name='train_seed_x_detokenizer_qwen_sdxl_ftunet_bs64_lr1e-4_wd5e-2_step20000'
OUTPUT_PATH=${WFS_ROOT}/training_logs/SEED-X/${exp_name}

mkdir -p $OUTPUT_PATH
torchrun --nproc_per_node=8 \
    ${PROJ_PATH}/src/train/train_detokenizer.py \
    --image_transform ${PROJ_PATH}/configs/processer/qwen_448_transform.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml \
    --diffusion_model_path /mnt/wfs/mmshanghai8wfssh/project_mm-base-vision-tj/huangzp/pretrained/stabilityai/stable-diffusion-xl-base-1.0 \
    --adapter_cfg_path configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize_fullft.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/qwen_vitg_448.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/sdxl_adapter_finetune.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name  ${exp_name} \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 2 \
    --mixed_precision no \
    --num_train_epochs 10 \
    --max_steps 20000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2_fp32.yaml \
    --resume_steps 6000 \
    --resume_from_checkpoint /mnt/wfs/mmshanghai8wfssh/project_mm-base-vision-tj/huangzp/training_logs/SEED-X/seed_x_detokenizer/checkpoint-6000 # 不要最后这个 pytorch_model/mp_rank_00_model_states.pt
    # --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2_offload.yaml \ 
    # --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_3.yaml \


echo '--------------------------'
echo main training task done
echo '--------------------------'
