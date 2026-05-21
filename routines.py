import zipfile
from pathlib import Path

# Конфигурация путей
source_dir = Path(r"C:\Users\admin\Downloads")
target_dir = Path(r"C:\Users\admin\PycharmProjects\NOF_analysis\LNOF_SynchronizedData")
# Маска для поиска архивов
file_pattern = "SynchronizedData-20260204T125735Z-3*.zip"


def extract_synchronized_data():
    # Создаем целевую папку, если она не существует
    target_dir.mkdir(parents=True, exist_ok=True)

    # Поиск всех архивов по шаблону
    zip_files = list(source_dir.glob(file_pattern))

    if not zip_files:
        print("Архивы по указанному шаблону не найдены.")
        return

    for zip_path in zip_files:
        print(f"Обработка архива: {zip_path.name}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Проверяем, находится ли файл внутри папки SynchronizedData/
                if member.filename.startswith("SynchronizedData/") and not member.is_dir():
                    # Убираем префикс 'SynchronizedData/', чтобы файлы легли прямо в целевую папку
                    filename = Path(member.filename).name
                    target_path = target_dir / filename

                    # Извлекаем содержимое файла
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

    print(f"\nГотово! Все файлы извлечены в: {target_dir}")

import numpy as np
import os


if __name__ == "__main__":
    # extract_synchronized_data()

    path = r'C:\Users\admin\PycharmProjects\NOF_analysis\LNOF_SynchronizedData\LNOF_J55_4D_aligned.npz'
    data = np.load(path)
    print(data.files)  # Покажет ключи массивов внутри архива (например, 'traces', 'timestamps')