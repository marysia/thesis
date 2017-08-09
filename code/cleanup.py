import glob
import os
from stat import ST_CTIME
import argparse
import time

logdir = '/home/marysia/thesis/logs'
folders = ['mnist', 'patches2d', 'patches3d']
keep_limit = 5

def cleanup_all():
    # clean up current log files
    logfiles = glob.glob(logdir + '/current*log')
    for logfile in logfiles:
        print('Removing %s.' % logfile)
        os.remove(logfile)

    # clean up folders
    for data_folder in folders:
        logfiles = [os.path.join(logdir, data_folder, f) for f in os.listdir(os.path.join(logdir, data_folder))]
        if len(logfiles) >= keep_limit + 1:
            # sort based on tiume
            logfiles = [(os.stat(path)[ST_CTIME], path) for path in logfiles]
            logfiles.sort()

            # remove old files
            for _, logfile in logfiles[:keep_limit + 1]:
                print('Removing %s.' % logfile)
                os.remove(logfile)

            # rename kept log files
            for i, (_, logfile) in enumerate(logfiles[keep_limit + 1:]):
                folder_structure = logfile.split('/')
                fname_split = folder_structure[-1].split('_')
                fname_split[0] = str(i + 1)
                new_fname = '_'.join(fname_split)
                folder_structure[-1] = new_fname
                dst = '/'.join(folder_structure)
                print('Renaming %s to %s' % (logfile, dst))
                os.rename(logfile, dst)

def cleanup_broken():
    broken_logfiles = glob.glob(logdir + '/*/broken*')
    if len(broken_logfiles) == 0:
        print('No broken logfiles.')
    else:
        for logfile in broken_logfiles:
            print('Removing %s.' % logfile)
            os.remove(logfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep', type=int, default=5)
    parser.add_argument('--broken', action='store_true')
    args = parser.parse_args()

    if args.broken:
        cleanup_broken()
    else:
        keep_limit = args.keep
        cleanup_all()
