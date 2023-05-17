# 1
# <<'COMMENT'
for idx in 1 2 3 4 5 6 7 8 9 10
do
    python3 train_custom_gcn.py --do_train \
                                --dataset="gcdc_clinton" \
                                --fold_id=${idx} \
                                --label_list="1, 2, 3" \
                                --k_node=4 \
                                --max_seq_length=512 \
                                --windom_size=8 \
                                --hidden_dim=240 \
                                --learning_rate=0.01 \
                                --num_train_epochs=160
done
# COMMENT
# 17