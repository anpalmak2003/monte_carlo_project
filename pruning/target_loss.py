from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from transformers import LlamaForCausalLM, AutoTokenizer
from accelerate import Accelerator
import os
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import random

# Инициализация Accelerator для распределенного вычисления
accelerator = Accelerator()
device = accelerator.device

model_id = '/home/jovyan/shares/SR006.nfs1/amaksimova/models/gigar_9b'
num_samples = 1000
max_length = 4096
stride = 512
dir_pth = '/home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/gigasets_full/'
output_file = '/home/jovyan/shares/SR006.nfs1/amaksimova/LLM-Shearing-main/losses_gigasets_preptrain.json'

# Загрузка модели и токенизатора
model = LlamaForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Подготовка модели для распределенного вычисления
model = accelerator.prepare(model)

def load_existing_results(output_file):
    """Загружает существующие результаты из файла или возвращает пустой словарь"""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return {}

def process_batch(batch, model, tokenizer, max_length, stride):
    texts = batch["text"]
    batch_loss = []
    batch_ppl = []
    
    for example in texts:
        encodings = tokenizer(example, return_tensors="pt", truncation=True, max_length=max_length)
        seq_len = encodings.input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if n_tokens > 0:
            avg_nll = nll_sum / n_tokens
            batch_loss.append(avg_nll.item())
            batch_ppl.append(torch.exp(avg_nll).item())

    return batch_loss, batch_ppl

def main():
    # Загружаем существующие результаты
    existing_results = load_existing_results(output_file)
    losses = existing_results.copy()
    
    for file_p in os.listdir(dir_pth):
        # Пропускаем, если уже есть результаты для этого датасета
        if file_p in existing_results:
            print(f"Skipping {file_p} - already processed")
            continue
            
        pth = Path(os.path.join(dir_pth, file_p))
        all_files = list(pth.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        
        if not all_files:
            print(f"No files found in {file_p}")
            continue
            
        fn = random.choice(all_files)
        print(f"Processing {fn}")

        try:
            # Чтение данных
            df = pd.read_json(fn, lines=True)
            df = pd.DataFrame(columns=['text'], data=df['text'])
            dataset = Dataset.from_pandas(df[:num_samples])
            
            # Создание DataLoader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            dataloader = accelerator.prepare(dataloader)
            
            all_loss = []
            all_ppl = []
            
            # Обработка батчей
            for batch in tqdm(dataloader, desc=f"Processing {file_p}"):
                batch_loss, batch_ppl = process_batch(batch, model, tokenizer, max_length, stride)
                all_loss.extend(batch_loss)
                all_ppl.extend(batch_ppl)
            
            # Сбор результатов со всех GPU
            all_loss = accelerator.gather(torch.tensor(all_loss, device=device)).cpu().numpy()
            all_ppl = accelerator.gather(torch.tensor(all_ppl, device=device)).cpu().numpy()
            
            # Расчет средних значений
            if accelerator.is_main_process:
                average_ppl = all_ppl.mean() if len(all_ppl) > 0 else float('inf')
                average_loss = all_loss.mean() if len(all_loss) > 0 else float('inf')
                losses[file_p] = [float(average_ppl), float(average_loss)]
                print(f"File: {file_p}, PPL: {average_ppl:.2f}, Loss: {average_loss:.2f}")
                
                # Сохраняем после каждого датасета на случай прерывания
                with open(output_file, 'w') as f:
                    json.dump(losses, f)
                    
        except Exception as e:
            print(f"Error processing {file_p}: {str(e)}")
            continue

if __name__ == "__main__":
    main()