import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import gc
from transformers import AutoTokenizer

# Конфигурация
input_dir = Path("/home/jovyan/shares/SR006.nfs1/amaksimova/data/cultura")
output_file = Path("/home/jovyan/shares/SR006.nfs1/amaksimova/data/cultura_ru_400M_tokens.txt")
batch_size = 50_000  # Размер батча для потокового чтения
target_token_count = 400_000_000  # Целевое количество токенов

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(
    "/home/jovyan/shares/SR006.nfs1/amaksimova/models/9b/GigaChat-2-Lite-128k-preview",
    trust_remote_code=True
)

# Собираем все файлы
parquet_files = sorted(input_dir.glob("ru_part_*.parquet"))

total_tokens = 0
file_counter = 0

with open(output_file, "w", encoding="utf-8") as f_out:
    # Прогресс-бар по файлам
    for file in tqdm(parquet_files, desc="Обработка файлов"):
        if total_tokens >= target_token_count:
            break
            
        # Используем PyArrow для потокового чтения
        table = pq.read_table(file)
        
        # Прогресс-бар по батчам
        for batch in tqdm(
            table.to_batches(max_chunksize=batch_size),
            desc=f"Обработка {file.name}",
            leave=False,
            total=table.num_rows // batch_size + 1
        ):
            if total_tokens >= target_token_count:
                break
                
            df = batch.to_pandas()
            for text in df["text"].dropna():
                # Токенизируем текст для подсчета токенов
                tokens = tokenizer.tokenize(text.strip())
                num_tokens = len(tokens)
                
                # Если добавление этого текста не превысит лимит
                if total_tokens + num_tokens <= target_token_count:
                    f_out.write(text.strip() + "\n")
                    total_tokens += num_tokens
                else:
                    # Добавляем часть текста, чтобы достичь точного количества
                    remaining_tokens = target_token_count - total_tokens
                    if remaining_tokens > 10:  # Добавляем только если осталось значительное количество
                        partial_text = tokenizer.convert_tokens_to_string(tokens[:remaining_tokens])
                        f_out.write(partial_text + "\n")
                        total_tokens += remaining_tokens
                    break
            
            del df, batch
            gc.collect()
        
        file_counter += 1

print(f"\nГотово! Обработано {file_counter} файлов.")
print(f"Итоговое количество токенов: {total_tokens:,}")
print(f"Данные сохранены в {output_file}")