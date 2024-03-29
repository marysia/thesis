import argparse
import glob
import os
from stat import ST_CTIME
from utils.config import LOGDIR
LOGDIR = '/home/marysia/thesis/logs'
folders = ['patches3d', 'composition']
keep_limit = 5


def cleanup_all():
    # clean up current log files
    logfiles = glob.glob(LOGDIR + '/running*log')
    if len(logfiles) > 0:
        print('Cleaning up current log files: ')
        for logfile in logfiles:
            print('\t -%s.' % logfile)
            os.remove(logfile)

    # remove pycharm files
    pycharm_files = glob.glob(LOGDIR + '/*/*pycharm*')
    for file in pycharm_files:
        os.remove(file)

    # clean up folders
    for data_folder in folders:
        logfiles = [os.path.join(LOGDIR, data_folder, f) for f in os.listdir(os.path.join(LOGDIR, data_folder))]
        if len(logfiles) >= keep_limit + 1:
            # sort based on time
            logfiles = [(os.stat(path)[ST_CTIME], path) for path in logfiles]
            logfiles.sort()

            # remove old files
            print('\nRemoving old files in %s: ' % data_folder)
            for _, logfile in logfiles[:keep_limit + 1]:
                print('\t -%s.' % logfile)
                os.remove(logfile)

            # rename kept log files
            print('\nRenaming kept log files in %s' % data_folder)
            for i, (_, logfile) in enumerate(logfiles[keep_limit + 1:]):
                folder_structure = logfile.split('/')
                fname_split = folder_structure[-1].split('_')
                fname_split[0] = str(i + 1)
                new_fname = '_'.join(fname_split)
                folder_structure[-1] = new_fname
                dst = '/'.join(folder_structure)
                print('\t - %s to %s' % (logfile, dst))
                os.rename(logfile, dst)

    # close all tmux windows
    try:
        os.system('tmux kill-server')
    except:
        raise Exception


def cleanup_broken():
    broken_logfiles = glob.glob(LOGDIR + '/*/broken*')
    if len(broken_logfiles) != 0:
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
        cleanup_broken()
        keep_limit = args.keep
        cleanup_all()
