MODEL_PATH=/home/jovyan/shares/SR006.nfs1/amaksimova/LLM-Shearing-main/llmshearing/models/llama2_9b_pruning_scaling_doremi_to1.3b_sl4096_lr5e-4_bs4_latest_ft40000ba/latest-rank0.pt
OUTPUT_PATH=/home/jovyan/shares/SR006.nfs1/amaksimova/LLM-Shearing-main/llmshearing/models/llama2_9b_pruning_scaling_doremi_to1.3b_sl4096_lr5e-4_bs4_latest_ft40000ba/cultura_1.3b
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=24
INTERMEDIATE_SIZE=5504
MODEL_NAME=Sheared-Llama-1.3B

python3 -m composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}
