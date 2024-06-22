
from multiprocessing import Pool
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put((self._guarded_task_generation(
        result._job, mpp.starmapstar, task_batches), result._set_length))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


# 好像函数里面不能塞torch的相关函数
# 缺省参数全部都在后面
def multi_func(func, max_process_num, total, desc, unzip, *args):
    # args is a tuple
    assert len(args) > 0 and len(args[0]) > 0
    from tqdm import tqdm
    multi_param = len(args) > 1
    args = list(zip(*args)) if multi_param else args[0]
    with Pool(max_process_num) as p:
        map_func = p.istarmap if multi_param else p.imap
        res = list(tqdm(map_func(func, args), total=total, desc=desc))
    return tuple([list(x) for x in zip(*res)]) if (unzip and isinstance(res[0], tuple)) else res
