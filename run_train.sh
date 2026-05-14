#!/bin/bash

# 默认设置：如果没指定，默认使用第 0 张卡
GPUS="4,6"
CONFIG="app/multimodal_jepa/configs/multimodal_base.yaml"

# 帮助信息函数
usage() {
    echo "用法: $0 [-g <gpu_ids>] [-c <config_path>]"
    echo "选项:"
    echo "  -g  指定要使用的显卡 ID，用逗号分隔 (例如: '0,1,2,3' 或 '6,7' 或 '6')"
    echo "  -c  指定配置文件路径 (默认: $CONFIG)"
    echo "  -h  显示此帮助信息"
    exit 1
}

# 解析命令行参数
while getopts "g:c:h" opt; do
    case $opt in
        g) GPUS="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# 将解析过的参数移除，保留可能传递给 train.py 的其他参数
shift $((OPTIND -1))

# 1. 限制对显卡的可见性
export CUDA_VISIBLE_DEVICES=$GPUS

# 2. 自动计算使用的显卡数量
# 原理：计算逗号的数量加 1 (例如 "6,7" 有1个逗号，说明是2张卡)
NUM_GPUS=$(echo $GPUS | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

echo "======================================================="
echo "🚀 正在启动分布式训练..."
echo "👉 可见显卡 (CUDA_VISIBLE_DEVICES) : $GPUS"
echo "👉 进程数量 (nproc_per_node)       : $NUM_GPUS"
echo "👉 配置文件                       : $CONFIG"
echo "======================================================="

# 3. 运行 torchrun
torchrun --nproc_per_node=$NUM_GPUS \
    app/multimodal_jepa/train.py \
    --config "$CONFIG" \
    "$@"  # 允许将额外参数透传给 train.py