#!/usr/bin/env bash

name='1024_44_netD32_2reso_only128loss_color_StyleGAN_AttnFuse_inputnoise128_instancenoise_PRtxtdomain'
dataset='flowers'
dir='/home/chendaiyuan/txt2imgGAN/Img_Models/'${name}_$dataset
mkdir -v $dir
CUDA_VISIBLE_DEVICES=${device} python3 train_PRtxtdomain_flowers.py \
                                --dataset $dataset \
                                --gpus ${device} \
                                --model_name ${name} \
                                --emb_dim 128 \
                                --instance_noise \
                                --hidden_dim 1024 \
                                --D_step_dim 32 \
                                --noise_dim 128 \
                                --batch_size 80 \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --server_port 12306 \
                                | tee $dir/'log.txt'

 # --instance_noise \
 # --label_smooth 0.1 \
 # --reuse_weights \
 # --load_from_epoch 800 \

# 1024_44_256-128-128-64-32_netD32_2reso_Segmentation_only128loss_PRtxtdomain
# 1024_44_256-128-128-64-32_netD32_2reso_only128loss_color_PRtxtdomain
# 512_44_modifiednetD32_2reso_Seg_only128loss_color_StyleGAN_All44AttnFuse_inputnoise128_PRtxtdomain_birds