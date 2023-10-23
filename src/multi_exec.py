# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:53:50 2023

@author: rabni
"""



import sicemachine as sm
from datetime import datetime, timedelta
from multiprocessing import set_start_method,get_context
import argparse
import pandas as pd
import sys 
import logging
import glob
import os
import time
from multiprocessing import set_start_method,get_context
import traceback

base_f  = os.path.abspath('..')

if not os.path.exists(base_f + os.sep + "logs"):
        os.makedirs(base_f + os.sep + "logs")

log_f = base_f + os.sep + "logs"
logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f'{log_f}' + os.sep + f'multiexec_{time.strftime("%Y_%m_%d",time.localtime())}.log'),
            logging.StreamHandler()
        ])

def parse_arguments():
        parser = argparse.ArgumentParser(description='Date range excicuteable for the CARRA2 Module')
        parser.add_argument("-st","--sday", type=str,help="Please input the start day")
        parser.add_argument("-en","--eday", type=str,help="Please input the end day")
        parser.add_argument("-c","--cores", type=int,default=4,help="Please input the number of cores you want to use")
        args = parser.parse_args()
        return args


def multiproc(day,model):
    logging.info(f'Processing {day}')
    classify = sm.ClassifierSICE()
    classify.predict_svm(day, model=model)
    return

if __name__ == "__main__":
    
    
    args = parse_arguments() 
    dates = pd.date_range(start=args.sday,end=args.eday).to_pydatetime().tolist()
    dates = [d.strftime("%Y-%m-%d") for d in dates]
    model = ['import' for i in range(len(dates))]
    
    set_start_method("spawn")
    
    
    logging.info('Parallelization Spawned')
    
    with get_context("spawn").Pool(args.cores) as p:     
            p.starmap(multiproc,zip(dates,model))
            p.close()
            p.join()
            logging.info("Done with multiprocessing")