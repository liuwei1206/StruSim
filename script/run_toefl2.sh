# 1
# <<'COMMENT'
for idx in 1 2 3 4 5
do
    python3 train_custom_gcn.py --do_train \
                                --dataset="toefl_p2" \
                                --fold_id=${idx} \
                                --label_list="low, medium, high" \
                                --k_node=5 \
                                --max_seq_length=1024 \
                                --windom_size=8 \
                                --hidden_dim=240 \
                                --learning_rate=0.05 \
                                --num_train_epochs=600
done
# COMMENT
# 17
