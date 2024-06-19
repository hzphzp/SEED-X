
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe



PROJ_PATH='.'
exp_name='seed_x_detokenizer'
OUTPUT_PATH=~/SEED-X/train_output/${exp_name}

mkdir -p $OUTPUT_PATH

export PYTHONPATH=$PROJ_PATH/proj/peft/src:$PYTHONPATH
wandb login 69672540eb30feaa4b6f38aa2b2aca504e0224ce
#torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20008 --node_rank=$INDEX \
torchrun --nproc_per_node=1 \
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
    --batch_size 50 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 2 \
    --mixed_precision fp16 \
    --num_train_epochs 10 \
    --max_steps 20000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2.yaml \ 
    # --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2_offload.yaml \ 
    # --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_3.yaml \


echo '--------------------------'
echo main training task done
echo '--------------------------'
