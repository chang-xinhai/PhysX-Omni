import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
import numpy as np
import ipdb
import re

def get_sorted_npy_list(folder_path):
    files = os.listdir(folder_path)
    
    pattern = re.compile(r'ind_(\d+)\.npy')
    
    npy_files = []
    for f in files:
        match = pattern.match(f)
        if match:
            npy_files.append((int(match.group(1)), f))
    
    npy_files.sort(key=lambda x: x[0])
    
    sorted_list = [f[1] for f in npy_files]
    count = len(sorted_list)
    
    return count, sorted_list

import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def decoder(name,basepath,imgpath):
    
    os.system('python decoder_each.py  --name {} --basepath {} --imgpath {}'.format(name,basepath,imgpath)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=500)
    parser.add_argument("--outputpath", type=str, default='./ours_demo')
    args = parser.parse_args()

    logger = get_logger(os.path.join('exp_2infer.log'),verbosity=1)
    basepath=args.outputpath
    namelist=os.listdir(basepath)
    logger.info('start')
    namelist=namelist[args.index*args.range:(args.index+1)*args.range]


    for name in namelist:
        

        logger.info('begin: '+name)
        condimgpath=os.path.join(basepath,name,"cond_img.png")
        objspath=os.path.join(basepath,name,'objs')
        qwenpath=os.path.join(basepath,name)
        n, sorted_files = get_sorted_npy_list(qwenpath)
        os.makedirs(os.path.join(objspath), exist_ok=True)

        if len(os.listdir(objspath))==n:
            logger.info('skip: '+name)
        else:
            
            decoder(name,basepath,condimgpath)
            n, sorted_files = get_sorted_npy_list(qwenpath)
            if len(os.listdir(objspath))==n:
                logger.info('success: '+name)
            else:
                logger.info('error: '+name)



