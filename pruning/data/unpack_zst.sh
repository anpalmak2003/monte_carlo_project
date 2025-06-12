#!/bin/bash

# Путь к целевой папке
target_dir="/home/jovyan/shares/SR006.nfs1/amaksimova/data/gigasets/gigasets_full2"

# Находим и распаковываем все .zst файлы
find "$target_dir" -type f -name "*.zst" -print0 | while IFS= read -r -d '' zst_file; do
    # Определяем путь для распакованного файла (убираем .zst)
    unpacked_file="${zst_file%.zst}"
    
    echo "Распаковываю: $zst_file"
    
    # Распаковываем с помощью zstd
    if zstd -d -f "$zst_file" -o "$unpacked_file"; then
        # Проверяем, что распакованный файл существует
        if [[ -f "$unpacked_file" ]]; then
            echo "Успешно распаковано в: $unpacked_file"
            # Удаляем оригинальный .zst файл
            rm "$zst_file"
            echo "Удален архив: $zst_file"
        else
            echo "Ошибка: распакованный файл не создан - $unpacked_file" >&2
        fi
    else
        echo "Ошибка при распаковке $zst_file" >&2
    fi
done

echo "Все .zst файлы обработаны"