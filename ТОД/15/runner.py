
# -*- coding: utf-8 -*-
import sys, os, shutil, time, multiprocessing, subprocess, pickle
import dask.dataframe as dd
from collections import defaultdict

def agg(dicts: list) -> dict:
    valid = [d for d in dicts if d is not None]
    d = defaultdict(lambda: {'sum': 0, 'count': 0})
    for d_ in valid:
        for i in d_:
            d[i]['sum'] += d_[i]['sum']
            d[i]['count'] += d_[i]['count']
    return d

def mean(data: dict) -> dict:
    return {i: data[i]['sum'] / data[i]['count'] for i in data if data[i]['count'] > 0}

def call_worker(file):
    cmd = [sys.executable, 'worker.py', file]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.buffer.write(proc.stderr)
        return None
    return pickle.loads(proc.stdout)

def run(K: int):
    base = r'Тема_Параллельные вычисления (1)/data'
    input = os.path.join(base, 'tag_nsteps.csv')
    odir = os.path.join(base, f'tag_steps_task7_{K}')

    if not os.path.exists(odir) or not os.listdir(odir):
        if os.path.exists(odir): shutil.rmtree(odir)
        os.makedirs(odir, exist_ok=True)
        df = dd.read_csv(input, encoding='utf-8')
        df = df.repartition(npartitions=K)
        df.to_csv(odir + '/tag_nsteps_*.csv', index=False, encoding='utf-8')

    files = [os.path.join(odir, f) for f in os.listdir(odir)]

    t0 = time.time()
    with multiprocessing.Pool() as pool:
        dicts = pool.map(call_worker, files)
    res = agg(dicts)
    mean(res)
    print(time.time() - t0)

if __name__ == '__main__':
    K = int(sys.argv[1])
    run(K)
