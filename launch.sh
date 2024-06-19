source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
wandb login 69672540eb30feaa4b6f38aa2b2aca504e0224ce

# the input scripts path
script_path=$1
# torchrun --nproc-per-node 8 --master-port 11111 $script_path
python $script_path