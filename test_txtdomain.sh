# test 128
name='1024_44_netD32_2reso_only128loss_color_StyleGAN_AttnFuse_inputnoise128_PRtxtdomain_coco'
CUDA_VISIBLE_DEVICES=${device} python3 test_2txtdomain.py \
                                    --load_from_epoch 800 \
                                    --dataset coco \
                                    --model_name ${name} \
                                    --emb_dim 128 \
                                    --hidden_dim 1024 \
                                    --noise_dim 128 \
                                    --test_sample_num 1 \
                                    --sampling_10 \
                                    --save_visual_results \
                                    --server_port 12315 \
                                    --batch_size 60
