NUM_FILES=$(wc -l < jsonl_list.txt)
echo "Total files: $NUM_FILES"
start_time=$SECONDS
for ((i=0; i<NUM_FILES; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python tokenize_paralel.py --target_dir /home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/gigasets_full_tokenized --raw_dir /home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/wikipedia_sharded
done

elapsed_time=$(( SECONDS - start_time ))
echo "Время выполнения: $elapsed_time секунд"
