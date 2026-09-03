
# -*- coding: utf-8 -*-
import dask
import dask.dataframe as dd
import pickle
import sys

def process(file: str) -> dict:
    with dask.config.set(scheduler='single-threaded'):
        df = dd.read_csv(file, encoding='utf-8')
        agg = df.groupby('tags').agg({'n_steps': ['sum', 'count']})
        res = agg.compute()
        res.columns = res.columns.droplevel(0)
        return res.to_dict(orient='index')

if __name__ == '__main__':
    file_path = sys.argv[1]
    result = process(file_path)
    pickle.dump(result, sys.stdout.buffer)
