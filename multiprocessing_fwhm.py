from multiprocessing import Lock, Manager
from tqdm import tqdm
import numpy as np
import time
# m = Manager()
lock = Lock()

log_file_path = './logs/'
log_name = 'testfunc.log'


def s_print(lock, *a, **b):
    """Thread safe print function"""
    with lock:
        print(*a, **b)
    time.sleep(1)

with lock:
    print('start of file')

import sys
# %load_ext autoreload
# %reload_ext autoreload
# %autoreload 2

MIPSGAL_DIR = '../../IRSA_Spitzer_Mipsgal/datasets/'
base_dir = './datasets/MG/'


# from modules.ProcessingTools.multiprocessing.makeFileSets import makeFileSets


def testfunc(a, log_queue):
    import logging
    import logging.handlers
    from os import path



    qh = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger(path.join(log_file_path, log_name))
    root_logger.addHandler(qh)
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(log_name).getChild(f'child_{i}') # This allows you to know what process you're in
    logger.setLevel(logging.INFO)

    logger.info('hello from inside process')

    # with lock:
    s_print('sprint')
    time.sleep(3)

    return True

if __name__ == '__main__':
    # makeFileSets(FILE_DIR)  
    # from multiprocess import pool
    from pathos.multiprocessing import ProcessingPool as Pool
    from multiprocess import process
    from multiprocessing import Manager

    # from multiprocessing import  get_context, pool, Lock
    from codetiming import Timer
    from modules.ajh_utils import handy_dandy as hd
    from modules.ProcessingTools.FitsProcessing import getFWHMsForFile
    import os
    from modules.ajh_utils import logger as lg 


    logger = lg.listener_configurer(log_file_path, log_name)
    manager = Manager()
    log_queue = manager.Queue()
    listener = process.BaseProcess(target=lg.listener_process, args=(log_queue, lg.listener_configurer, log_name))

    listener.start()
    
    all_files = os.listdir(base_dir)

    # sys.stdout.write('test')
    print('Start of __main__')

    n = 10
    timer = Timer(name="class")
    timer.start()
    # 
    results = []
    fwhms = []
    # with pool.Pool(processes=6) as p, tqdm(total=n) as  pbar:
    with Pool(processes=6) as p, tqdm(total=n) as  pbar:
    # with get_context('spawn').Pool(processes=6) as p:

        logger.info('start of iteration') 
        # func = lambda b : testfunc(base_dir, b)
        # results = p.imap_unordered(func, base_dir) 
        results = [ p.apipe(testfunc, f, log_queue) for f in range(0,10)]
        # results = [ p.apply_async(getFWHMsForFile, args=(base_dir, f,)) for f in all_files ]
        # results = [ p.apply_async(getFWHMsForFile, args=(base_dir, f,), callback=lambda _: pbar.update(1)) for f in all_files ]
        # results = [ p.apply_async(testfunc, args=(base_dir, f,), callback=lambda _: pbar.update(1)) for f in all_files ]



        for r in results:
            # print(r.get())
            print(r)
            # fwhms, dataset = r.get()
            # dataset.fwhm = fwhms
            # print(f'{fwhms = }')
        # #     # dataset.saveFileSet()
            # with open(f'./datasets/cutouts/dills/{dataset.source_filename}_training_testing_headers.dill', 'wb') as f:
            #     dill.dump(dataset, f, byref=True)

            

    timer.stop()
