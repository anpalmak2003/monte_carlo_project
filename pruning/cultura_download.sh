#!/bin/bash

# Директория для сохранения
OUTPUT_DIR="/home/jovyan/shares/SR006.nfs1/amaksimova/data/cultura"
mkdir -p "$OUTPUT_DIR"

# Базовый URL
BASE_URL="https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/ru/ru_part_"

# Скачиваем 200 файлов (от 00000 до 00199)
for i in {166..167}; do
    # Форматируем номер файла (5 цифр с ведущими нулями)
    FILE_NUM=$(printf "%05d" $i)
    FILE_URL="${BASE_URL}${FILE_NUM}.parquet"
    OUTPUT_FILE="${OUTPUT_DIR}/ru_part_${FILE_NUM}.parquet"
    
    echo "Скачиваю файл $FILE_NUM..."
    wget -q --show-progress --header="Authorization: Bearer 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" -O "$OUTPUT_FILE" "$FILE_URL"
    
    # Проверяем успешность загрузки
    if [ $? -eq 0 ]; then
        echo "Файл $FILE_NUM успешно сохранён в $OUTPUT_FILE"
    else
        echo "Ошибка при скачивании файла $FILE_NUM"
    fi
done

echo "Загрузка завершена. Файлы сохранены в $OUTPUT_DIR"