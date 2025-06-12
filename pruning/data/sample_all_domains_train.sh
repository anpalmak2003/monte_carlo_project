NUM_DOMAINS=20
echo "Total domains: $NUM_DOMAINS"

for ((i=0; i<NUM_DOMAINS; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python sample_train.py --target_dir /home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/gigasets_full_tokenized_mds --tokenized_dir /home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/gigasets_full_tokenized
done
