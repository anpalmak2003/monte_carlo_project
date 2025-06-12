from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os
import json
import numpy as np
import argparse
from multiprocessing import Pool
import zstandard as zstd
import io
from pathlib import Path

def stream_zst_file(file_path):
    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_reader:
                if line.strip():
                    yield json.loads(line)

def process_file(file_info):
    file_name, args = file_info
    file_path = os.path.join(args.raw_dir, file_name)
    target_name = os.path.join(args.target_dir, os.path.splitext(file_name)[0] + ".npy")

    # Проверяем существование целевого файла
    if os.path.exists(target_name):
        print(f"Skipping (already exists): {target_name}")
        return file_name

    print(f"Processing {file_path} -> {target_name}")

    # Создаем целевую директорию
    target_folder = os.path.dirname(target_name)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    try:
        # Загрузка и обработка данных
        if file_path.endswith('.zst'):
            lines = stream_zst_file(file_path)
        else:
            lines = open(file_path, 'r', encoding='utf-8')

        buffer = []
        data = []

        for line in tqdm(lines, desc=f"Processing {file_name}"):
            try:
                item = json.loads(line)
                tokens = buffer + tok.encode(item["text"], padding=True, truncation=True)
                buffer = []

                for start_id in range(0, len(tokens), args.seq_length):
                    if start_id + args.seq_length < len(tokens):
                        data.append(tokens[start_id:start_id+args.seq_length])
                    else:
                        buffer = tokens[start_id:]
                        break
            except json.JSONDecodeError:
                print(f"JSON decode error in {file_name}")
                continue
            except KeyError:
                print(f"Missing 'text' field in {file_name}")
                continue

        # Сохранение numpy array
        if data:
            data = np.array(np.stack(data), dtype=np.uint16)
            np.save(target_name, data)
            print(f"Saved to {target_name}")
        else:
            print(f"No data to save for {file_name}")

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        if os.path.exists(target_name):
            os.remove(target_name)  # Удаляем частично обработанный файл

    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default='/home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets',
                       type=str, help="Directory for raw data")
    parser.add_argument("--target_dir", default='/home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets_tokenized',
                       type=str, help="Target directory to save tokenized numpy")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of parallel processes")
    args = parser.parse_args()

    # Загрузка токенизатора
    print("Loading tokenizer...")
    global tok
    tok = AutoTokenizer.from_pretrained('/home/jovyan/shares/SR006.nfs1/amaksimova/models/GigaChat-2-Lite-128k-preview_mh',
                                      add_eos_token=True)
    tok.sep_token = tok.eos_token
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    print("Tokenizer loaded")

    # Получение списка файлов для обработки
    with open("/home/jovyan/shares/SR006.nfs1/amaksimova/LLM-Shearing-main/llmshearing/data/jsonl_list.txt", 'r', encoding='utf-8') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # Подготовка задач для параллельной обработки
    tasks = [(file_name, args) for file_name in file_list]

    # Обработка файлов параллельно
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_file, tasks), total=len(file_list)))

    print("\nProcessing complete:")
    for res in results:
        if res:  # Пропускаем None (уже обработанные файлы)
            print(f"- {res}")