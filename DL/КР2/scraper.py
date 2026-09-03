import os
import re
import time
import hashlib
import requests
import random
import shutil
from pathlib import Path
from urllib.parse import unquote

# --- КОНФИГУРАЦИЯ ---
BASE_DIR = Path("dataset_dumplings")
HISTORY_URLS_FILE = BASE_DIR / "history_urls.txt"
HISTORY_HASHES_FILE = BASE_DIR / "history_hashes.txt"

# Имитация браузера
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
}

# Расширенные запросы для максимального охвата
TAXONOMY = {
    'pelmeni':  ['pelmeni', 'pelmeni dumplings', 'siberian pelmeni', 'russian dumplings', 'пельмени домашние', 'пельмени в бульоне', 'raw pelmeni', 'cooked pelmeni'],
    'khinkali': ['khinkali', 'georgian khinkali', 'khinkali dumplings', 'хинкали', 'хинкали с зеленью', 'хинкали в разрезе', 'big khinkali'],
    'gyoza':    ['gyoza', 'gyoza dumplings', 'fried gyoza', 'japanese pan fried dumplings', 'yaki gyoza', 'steamed gyoza', 'гёдза'],
    'manti':    ['manti', 'manti dumplings', 'uzbek manti', 'turkish manti', 'манты', 'манты в мантоварке', 'манты с тыквой', 'big manti']
}

class DatasetMiner:
    def __init__(self):
        self.prepare_dirs()
        self.visited_urls = self.load_set(HISTORY_URLS_FILE)
        self.visited_hashes = self.load_set(HISTORY_HASHES_FILE)
        self.session = requests.Session()
        self.session.trust_env = False
        
        self.session.headers.update(HEADERS)


    def prepare_dirs(self):
        if not BASE_DIR.exists():
            BASE_DIR.mkdir()
        for label in TAXONOMY.keys():
            (BASE_DIR / label).mkdir(exist_ok=True)

    def load_set(self, filepath):
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        return set()

    def append_to_file(self, filepath, item):
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"{item}\n")

    def get_md5(self, content):
        return hashlib.md5(content).hexdigest()

    def download_image(self, url, label):
        # 1. Быстрая проверка URL
        if url in self.visited_urls:
            return False
        
        try:
            r = self.session.get(url, timeout=5)
            if r.status_code != 200:
                return False
            
            content = r.content
            
            # 2. Проверка содержимого (дубликат картинки)
            img_hash = self.get_md5(content)
            if img_hash in self.visited_hashes:
                # Помечаем URL как посещенный, даже если картинка дубль, чтобы не качать снова
                self.visited_urls.add(url)
                self.append_to_file(HISTORY_URLS_FILE, url)
                return False

            # Сохранение
            filename = f"{label}_{int(time.time())}_{random.randint(1000,9999)}.jpg"
            save_path = BASE_DIR / label / filename
            
            with open(save_path, 'wb') as f:
                f.write(content)

            # Обновление баз
            self.visited_urls.add(url)
            self.append_to_file(HISTORY_URLS_FILE, url)
            self.visited_hashes.add(img_hash)
            self.append_to_file(HISTORY_HASHES_FILE, img_hash)
            
            return True

        except Exception:
            return False

    def bing_search_iter(self, query):
        """Генератор ссылок с Bing"""
        base_url = "https://www.bing.com/images/async"
        count = 35
        first = 0
        
        while True:
            params = {
                "q": query,
                "first": first,
                "count": count,
                "adlt": "off",
                "qft": "" # filter (можно добавить +filterui:photo-photo)
            }
            try:
                resp = self.session.get(base_url, params=params)
                if resp.status_code != 200:
                    break
                
                # Regex для поиска прямых ссылок (murl) внутри HTML
                # Bing прячет их в json-подобной структуре murl:"..."
                links = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
                
                if not links:
                    break
                    
                for link in links:
                    yield unquote(link) # Декодируем URL
                    
                first += count
                time.sleep(1.5) # Вежливость к серверу
                
                # Если Bing начинает отдавать мусор или повторы, это контролируется уровнем выше
                if first > 1000: # Ограничение глубины одного запроса
                    break
                    
            except Exception as e:
                print(f"Error fetching page: {e}")
                break

    def run(self):
        print(f"🚀 Starting Infinite Miner...")
        print(f"📂 Dataset: {BASE_DIR}")
        print(f"📚 Known URLs: {len(self.visited_urls)}")
        print(f"🖼 Known Images: {len(self.visited_hashes)}")
        
        while True:
            # Перемешиваем ключи и запросы для равномерного наполнения
            labels = list(TAXONOMY.keys())
            random.shuffle(labels)
            
            for label in labels:
                queries = TAXONOMY[label]
                random.shuffle(queries)
                
                for query in queries:
                    print(f"\n🔎 Searching: '{query}' [{label.upper()}]")
                    new_count = 0
                    
                    for link in self.bing_search_iter(query):
                        if self.download_image(link, label):
                            new_count += 1
                            print(f"   ✅ Saved ({new_count}): ...{link[-30:]}")
                        else:
                            # print(f"   Skip") # Раскомментировать для дебага
                            pass
                            
                        # Если скачали достаточно много за один заход по запросу, можно переключиться
                        if new_count >= 50:
                            print("   ☕ Switching query to vary dataset...")
                            break
                            
                    if new_count == 0:
                        print("   ❌ No new images found for this query chunk.")

# --- ЗАПУСК ---
if __name__ == "__main__":
    miner = DatasetMiner()
    try:
        miner.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user. Data is safe.")