import os
import sys
import getopt
import psutil
import subprocess
import signal
import time


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
    pids = psutil.pids()
    processes = []
    for pid in pids:
        try:
            process = psutil.Process(pid)
            if 'python' in process.name():
                cmdline = process.cmdline()
                if len(cmdline) >= 2 and not '/usr/bin/python' in cmdline[0] and not 'control.py' in cmdline:
                    scriptname = [elem for elem in cmdline if '.py' in elem]
                    if len(scriptname) > 0:
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
            print('user: ' + str(user) + '\tpid: ' + str(pid) + '\tscript: ' + file)
    else:
        print('--- No running python processes ---')


def start(script, verbose, params):
    ''' Start script in background. '''
    # check if already running
    process = psutil.Process(os.getpid())
    python = process.name()
    user = process.username()

    processes = find_python_processes()

    scripts = [elem[1] for elem in processes if elem[2] == user]
    if script.split('/')[-1] in scripts:
        print('%s is already running.' % script)
    else:
        print('Starting %s' % script)

        if not verbose:
            splitted = script.split('/')
            logname = splitted[-1].replace('.py', '.log')
            log = os.path.join('/'.join(splitted[:-1]), 'log', logname)
            cmd = 'setsid ' + python + ' ' + script + ' ' + params + ' &>' + log
        else:
            cmd = python + ' ' + script + ' ' + params + ' &'

        print(cmd)
        subprocess.Popen(cmd, shell=True)

# send a terminating signal (sigint or sigkill) to the provided script
def end(script, force):
    '''
    Send a terminating signal (sigint or sigkill) to the provided script.
    * script: script file name. do not include folder.
    * force: false for sigint, true for sigkill
    '''
    user = psutil.Process(os.getpid()).username()
    processes = find_python_processes()
    processes = [elem for elem in processes if (elem[2] == user)]
    processes = [elem for elem in processes if (elem[1] == script and script != 'all')]

    if len(processes) > 0:
        for pid, file, user in processes:
            print('Stopping %s with pid %s' % (file, str(pid)))
            if force:
                cmd = 'kill -9 ' + str(pid)
            else:
                cmd = 'kill -2 ' + str(pid)
            subprocess.Popen(cmd, shell=True)
    else:
        print('%s was not running.' % script)


def main(argv):
    '''
    Handle command line arguments.
    '''
    try:
        opts, args = getopt.getopt(argv, "hfrvs:e:", ["start=", "end="])
    except getopt.GetoptError:
        print('Incorrect use of flags. Use -h.')
        sys.exit(2)

    force = True if ('-f', '') in opts else False
    verbose = True if ('-v', '') in opts else False
    for opt, arg in opts:
        if opt == '-h':
            print(
            'python control.py \n \t * -h for help \n \t * -r for running processes \n \t * -s <script> for starting (add -v print text to commandline) \n \t * -e <script> for ending (add -f to force)')
            sys.exit()
        elif opt == '-r':
            running()
            sys.exit()
        elif opt in ("-s", "--start"):
            if '.py' in arg:
                params = ''
                parameters = argv[-1].strip()
                if parameters[0] == '[' and parameters[-1] == ']':
                    params = parameters[1:-1]
                start(arg, verbose, params)
            else:
                print(arg + ' is not a python file.')
            sys.exit()
        elif opt in ("-e", "--end"):
            if '.py' in arg:
                end(arg, force)
            else:
                print(arg + ' is not a python file.')
            sys.exit()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(['-r'])