#!/usr/bin/env python

import datetime
import glob
import time
import os

def do_cleanup(num_to_keep, glob_arg):
    files_list = glob.glob(glob_arg)
    files_list.sort(key=os.path.getmtime)
    for a_file in files_list[:-num_to_keep]:
        print("[{}]: removing file '{}' (not among {} newest)".format(datetime.datetime.now(), a_file, num_to_keep))
        os.remove(a_file)

if __name__ == '__main__':
    num_to_keep = 10
    glob_arg = '/pod_data/*.egg'
    while True:
        do_cleanup(num_to_keep, glob_arg)
        time.sleep(20)
