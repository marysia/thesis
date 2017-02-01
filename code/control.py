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
        '''
        Set terminate to true.
        '''
        self.terminate = True

def find_python_processes():
    '''
    Return all (pid, file, user) combinations of currently running python processes.
    '''
    pids = psutil.pids()    # note: old version of psutil installed, newer = psutil.pids()
    processes = []
    for pid in pids:
        try:
            process = psutil.Process(pid)

            if 'python' in process.name():
                cmdline = process.cmdline()
                if len(cmdline) == 2:
                    scriptname = [elem for elem in cmdline if '.py' in elem]
                    scriptname = str(scriptname[0]).split('/')[-1]

                    processes.append((pid, scriptname, process.username()))
        except psutil.NoSuchProcess:
            pass
    return processes

def running():
    '''
    Return verbose information of running processes.
    '''
    processes = find_python_processes()
    if len(processes) > 0:
        print('--- list of running python processes --- ')
        for pid, file, user in processes:
            print('user: ' + str(user) + '\tpid: ' + str(pid) +'\tscript: ' + file )
    else:
        print('--- No running python processes ---')



def start(script):
    ''' Start script in background. '''
    # check if already running
    if running(script):
        print('%s is already running.' % script)
    else:
        print('Starting %s' % script)
        cmd = 'python ' + script + ' &'
        subprocess.Popen(cmd, shell=True)

# send a terminating signal (sigint or sigkill) to the provided script
def end(script, force=False):
    '''
    Send a terminating signal (sigint or sigkill) to the provided script.
    * script: script file name. do not include folder.
    * force: false for sigint, true for sigkill
    '''
    processes = find_python_processes()
    processes = [elem for elem in processes if (elem[1] == script and script != 'all')]

    if len(processes) > 0:
        for pid, file, user in processes:
            print('Stopping %s with pid %s' % (file, str(pid)))
            if force:
                cmd = 'kill -9 ' + str(pid)
            else:
                cmd = 'kill -2 ' + str(pid)
            subprocess.Popen(cmd, shell=True)

            current_processes = find_python_processes()
            if (pid, file, user) not in current_processes:
                print('Stopped %s with pid %s succesfully.' % (file, str(pid)))
            else:
                print('Stopping %s with pid %s was not successful.' % (file, str(pid)))
    else:
        print('%s was not running.' % script)




def main(argv):
    '''
    Handle command line arguments.
    '''
    try:
        opts, args = getopt.getopt(argv, "hfrs:e:", ["start=", "end="])
    except getopt.GetoptError:
        print 'fail.'
        sys.exit(2)

    force = True if ('-f', '') in opts else False
    for opt, arg in opts:
        if opt == '-h':
            print 'use.'
            sys.exit()
        elif opt == '-r':
            running()
            sys.exit()
        elif opt in ("-s", "--start"):
            if '.py' in arg:
                start(arg)
            else:
                print arg + ' is not a python file.'
            sys.exit()
        elif opt in ("-e", "--end"):
            if '.py' in arg:
                end(arg, force)
            else:
                print arg + ' is not a python file.'
            sys.exit()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
