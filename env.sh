source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe

cd ~
ln -s /mnt/wfs/mmshanghai8wfssh/project_mm-base-vision-tj/huangzp data

# this file's dir
current_dir=$(cd "$(dirname "$0")";pwd)
cd $current_dir
ln -s ~/data/pretrained/SEED-X/ pretrained
# 注意这里需要使用腾讯的镜像, 不需要开proxy, 速度非常快
pip install transformers_stream_generator -i https://mirrors.tencent.com/pypi/simple/
pip install hydra-core -i https://mirrors.tencent.com/pypi/simple/
pip install pyrootutils -i https://mirrors.tencent.com/pypi/simple/
pip install torchdata -i https://mirrors.tencent.com/pypi/simple/
pip install wandb -i https://mirrors.tencent.com/pypi/simple/
wandb login 69672540eb30feaa4b6f38aa2b2aca504e0224ce