import argparse
import json
import os
import sys
from pathlib import Path

# Папки, обход которых запрещен
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".ipynb_checkpoints",
}

# Системные и служебные файлы, которые запрещено трогать
IGNORED_FILES = {
    ".git",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".gitkeep",
    ".mailmap",
    "dockerfile",
    "containerfile",
    "makefile",
    "gnumakefile",
    "license",
    "licence",
    "copying",
    "unlicense",
    "readme",
    "changelog",
    "contributing",
    "authors",
    "procfile",
    "gemfile",
    "rakefile",
    "vagrantfile",
    ".ds_store",
    ".env",
    ".editorconfig",
}

# Известные расширения, не являющиеся ноутбуками (для исключения лишнего чтения с диска)
KNOWN_NON_NOTEBOOK_EXTENSIONS = {
    ".py", ".pyw", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java",
    ".html", ".htm", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tiff",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    ".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls",
    ".csv", ".tsv", ".parquet", ".feather", ".h5", ".hdf5",
    ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".bin",
    ".exe", ".dll", ".so", ".dylib", ".sh", ".bat", ".cmd", ".ps1",
    ".mp3", ".wav", ".mp4", ".avi", ".mkv", ".mov",
}

# Ошибочные суффиксы, которые браузеры или редакторы иногда дописывают после .ipynb
TRAILING_DOWNLOAD_EXTENSIONS = (
    ".ipynb.txt",
    ".ipynb.json",
    ".ipynb.html",
    ".ipynb.bak",
    ".ipynb.tmp",
    ".ipynb.temp",
)


def is_jupyter_notebook(file_path: Path) -> bool:
    """
    Проверяет, является ли файл валидным документом формата Jupyter Notebook.
    Выполняет быстрое чтение первых байтов и последующий синтаксический разбор JSON.
    """
    try:
        # Быстрая проверка первого символа без загрузки всего файла в память
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if not chunk:
                return False
            # Пропуск UTF-8 BOM, если присутствует
            if chunk.startswith(b"\xef\xbb\xbf"):
                chunk = chunk[3:]
            chunk = chunk.strip()
            if not chunk or not chunk.startswith(b"{"):
                return False

        # Полная проверка структуры документа
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False

        has_cells = "cells" in data or "worksheets" in data
        has_nbformat = "nbformat" in data and isinstance(data["nbformat"], int)
        has_metadata = "metadata" in data and isinstance(data["metadata"], dict)

        return has_cells and has_nbformat and has_metadata

    except (json.JSONDecodeError, UnicodeDecodeError, PermissionError, OSError):
        return False


def get_target_filename(name: str) -> str:
    """
    Вычисляет целевое имя файла с расширением .ipynb.
    Сохраняет цифры версий (например, 'ML 3.3' -> 'ML 3.3.ipynb').
    """
    lower = name.lower()

    # Срезает ошибочно добавленные расширения после .ipynb
    for ext in TRAILING_DOWNLOAD_EXTENSIONS:
        if lower.endswith(ext):
            return name[: len(name) - len(ext) + len(".ipynb")]

    # Во всех остальных случаях добавляет .ipynb в конец исходного имени
    return f"{name}.ipynb"


def process_directory(root_dir: Path, dry_run: bool = False) -> None:
    """
    Рекурсивно обходит указанную директорию и переименовывает файлы.
    """
    if not root_dir.is_dir():
        print(f"Ошибка: путь '{root_dir}' не является директорией.")
        sys.exit(1)

    print(f"Начало сканирования каталога: {root_dir}")
    print(f"Режим предпросмотра (dry-run): {'ВКЛЮЧЕН' if dry_run else 'ВЫКЛЮЧЕН'}\n")

    scanned_files_count = 0
    renamed_files_count = 0
    skipped_existing_count = 0

    for current_root, dirs, files in os.walk(root_dir):
        # Исключение служебных каталогов из дальнейшего обхода
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        current_path = Path(current_root)

        for filename in files:
            scanned_files_count += 1
            file_path = current_path / filename

            # 1. Проверка системного черного списка
            if filename.lower() in IGNORED_FILES:
                continue

            # 2. Пропуск файлов, у которых уже установлено расширение .ipynb
            if filename.lower().endswith(".ipynb"):
                continue

            # 3. Пропуск известных форматов данных и исходного кода
            suffix = file_path.suffix.lower()
            if suffix in KNOWN_NON_NOTEBOOK_EXTENSIONS:
                continue

            # 4. Проверка содержимого файла на соответствие формату Jupyter Notebook
            if not is_jupyter_notebook(file_path):
                continue

            # 5. Определение нового имени
            new_filename = get_target_filename(filename)
            new_file_path = current_path / new_filename

            # 6. Проверка на коллизию имен
            if new_file_path.exists():
                print(
                    f"[ПРОПУСК] Файл уже существует: {new_file_path.relative_to(root_dir)}"
                )
                skipped_existing_count += 1
                continue

            # 7. Переименование
            if dry_run:
                print(
                    f"[DRY-RUN] {file_path.relative_to(root_dir)} -> {new_filename}"
                )
            else:
                try:
                    file_path.rename(new_file_path)
                    print(
                        f"[УСПЕХ]   {file_path.relative_to(root_dir)} -> {new_filename}"
                    )
                except OSError as e:
                    print(
                        f"[ОШИБКА]  Не удалось переименовать {file_path}: {e}"
                    )
                    continue

            renamed_files_count += 1

    print("\n" + "=" * 50)
    print("Итоги работы:")
    print(f"  Всего просмотрено файлов:   {scanned_files_count}")
    print(f"  Файлов к переименованию:     {renamed_files_count}")
    print(f"  Пропущено из-за коллизий:   {skipped_existing_count}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Поиск и переименование файлов без расширения или с поврежденным расширением в .ipynb"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Путь к целевой папке (если не указан, запрашивается ввод)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Выполнить сканирование без физического переименования файлов на диске",
    )

    args = parser.parse_args()

    target_folder = args.folder
    if not target_folder:
        target_folder = input("Введите путь к папке: ").strip()
        # Удаление возможных кавычек, добавляемых проводником при копировании пути
        target_folder = target_folder.strip('"').strip("'")

    if not target_folder:
        print("Ошибка: путь не был указан.")
        sys.exit(1)

    target_path = Path(target_folder).expanduser().resolve()
    process_directory(target_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()