import os
import sys
import getopt
import psutil
import subprocess
import signal


# Class to interpret interrupt signal
class ProgramEnder:
    '''
    Class to interpret interrupt/hangup signal.
    Used to gracefully exit running program.
    '''
    def __init__(self):
        self.terminate = False
        signal.signal(signal.SIGINT, self.set_exit)
        signal.signal(signal.SIGHUP, self.set_exit)

    def set_exit(self, signum, frame):
        self.terminate = True

# return all (pid, filename) combinations of currently running python processes
def find_python_processes(target=None):
    if target is not None:
        target = target.split('/')[-1]
    pids = psutil.get_pid_list()    # note: old version of psutil installed, newer = psutil.pids()
    processes = []
    for pid in pids:
        try:
            cmdline = psutil.Process(pid).cmdline()
            if len(cmdline) > 1:
                if ((target is None and 'python3.4' in cmdline or 'python' in cmdline or 'python3' in cmdline) or \
                            (target is not None and target in cmdline[1])) \
                        and not 'control.py' in cmdline[1]:
                    scriptname = [elem for elem in cmdline if '.py' in elem]
                    scriptname = scriptname[0].split('/')[-1]  # take only the filename, ignore the directory
                    processes.append((pid, scriptname))
        except psutil.NoSuchProcess:
            pass
    return processes

# return whether or not the provided filename is currently running
def running(script, info=False):
    filename = script.split('/')[-1]
    processes = find_python_processes()
    if info:
        if len(processes) > 0:
            print('--- list of running python processes --- ')
            for elem in processes:
                print('pid: ', elem[0], ' script: ', elem[1])
        else:
            print('--- No running python processes ---')

    running_files = [file for pid, file in processes]

    return filename in running_files


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hfrs:e:", ["start=", "end="])
    except getopt.GetoptError:
        print 'fail.'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'use.'
            sys.exit()
        elif opt == '-r':
            running('', info=True)
            sys.exit()
        elif opt in ("-s", "--start"):
            print 'Start process ' + arg
            sys.exit()
        elif opt in ("-e", "--end"):
            print 'End process ' + arg
            sys.exit()

if __name__ == "__main__":
   main(['-r'])
