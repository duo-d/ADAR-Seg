ngpu=4
batch_size=4
tag=val_dsnet_slurm_dist

srun -p dsta \
    --job-name=val_dsnet \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 12346 \
        --batch_size ${batch_size} \
        --config cfgs/release/dsnet.yaml \
        --pretrained_ckpt pretrained_weight/dsnet_pretrain_pq_0.577.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance \
        --onlyval \
        # --saveval # if you want to save the predictions of the validation set, uncomment this line
