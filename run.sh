args=(
    --dataset $3
    --ckpt vit_21k 
    --method efficientfsl 
    --exp_name $2
    --shot $4 
)
CUDA_VISIBLE_DEVICES=$1 python3 trainfs.py "${args[@]}"

# bash run.sh 1 demo mini-ImageNet 1