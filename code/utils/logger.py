import datetime
import time
import glob
import os
import random
import shutil
import string


class Logger():
    def __init__(self, args, logdir, logfolder, logname='log', time=False, depth=2, revision=False):
        '''
        Initialisiation of logger class.
        * args: list of arguments (sys.argv) used to run the file from cmdline
        * time: boolean global option to log time for each logging entry
        * depth: int to determine folder depth and file naming convention (only relevant if revision=False)
        * revision: boolean to circumvent date/timestamp file and folder naming convention. 
        Calls for function to create directory based on time stamp, create the log file
        and copy the running file to directory. 
        '''
        self.logdir = logdir
        self.logfolder = logfolder
        self.logname = logname
        self.time = time
        self.depth = depth
        self.revision = revision
        self.prefix = ''  # can be altered in self.create_directory()
        self.discard = args.discard

        self.runid = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))

        # define log paths
        self.log_path = self.current_log_name()
        self.latest_logpath = os.path.join(self.logdir, 'latest.log')

        self.cleanup()
        if not self.discard:
            with open(self.log_path, 'w') as f:
                f.write('----- LOG -----\n')
                f.write(str(args))



    # --- writing functions --- #            
    def write_to_file(self, text, code, time=False):
        ''' 
        Generic write to file function. 
        * text: the string to be appended to the file.
        * code: ERROR, INFO, RESULT etc. 
        * time: extra parameter to ensure time is written along with the
                message, even if self.time is false.
        '''
        # print to console as well.
        print(text)
        if not self.discard:
            with open(self.log_path, 'a') as f:
                f.write('\n')
                if time or self.time:
                    f.write(self.get_time())
                f.write(code + str(text))

    def result(self, text, time=True):
        ''' Writes result to log file. '''
        self.write_to_file(text, '[RESULT] \t', time)

    def info(self, text, time=False):
        ''' Writes result to log file. '''
        self.write_to_file(text, '', time)
        # self.write_to_file(text, '[INFO] \t \t', time)

    def error(self, text, time=True):
        ''' Writes result to log file. '''
        self.write_to_file(text, '[ERROR] \t', time)

    # --- helper functions --- #
    def cleanup(self):
        if 'pycharm' in self.logname:
            pycharm_files = glob.glob(os.path.join(self.logdir, '*pycharm*'))
            for pycharm_file in pycharm_files:
                os.remove(pycharm_file)

    def finalize(self, termination, exception):
        ''' Copies current running file to logdir/datafolder/logname.log and logdir/latest.log
        and removes logdir/current.log'''
        if not self.discard:
            if not 'pycharm' in self.logname:
                broken = True if termination or exception else False
                shutil.copy(self.log_path, self.final_log_name(self.logfolder, self.logname, broken))
                shutil.copy(self.log_path, self.latest_logpath)
            os.remove(self.log_path)

    def get_time(self):
        ''' Returns timestamp in appropriate format. '''
        t = time.time() + (2 * 3600)  # system time is two hours behind; adjust.
        timestamp = datetime.datetime.fromtimestamp(t).strftime(' \t [%Y-%m-%d %H:%M:%S] ')
        return timestamp

    # --- naming functions --- #
    def current_log_name(self):
        '''
        Naming convention: current.log, else current_2.log/current_3.log, etc.
        Enables multiple simultaneous runs.
        '''


        for i in xrange(1, 10):
            files = glob.glob(os.path.join(self.logdir, 'running_%d*' % i))
            if len(files) == 0:
                filepath = os.path.join(self.logdir, 'running_%d_%s.log' % (i, self.logname))
                return filepath

        raise Exception('Too many current files running, error.')


    def final_log_name(self, logfolder, logname, broken):
        '''
        Naming convention: logdir/datafolder/number_logname.log for healthy logs,
        logdir/datafolder/broken_number_logname.log for runs that were terminated during the run and could not be completed.
        '''
        log_folder = os.path.join(self.logdir, logfolder)
        list_of_logs = glob.glob(log_folder + '/*')

        if broken:
            broken_logs = [elem for elem in list_of_logs if 'broken' in elem]
            name = 'broken_%d_%s_%s.log' % (len(broken_logs) + 1, self.runid, logname)
        else:
            healthy_logs = [elem for elem in list_of_logs if not 'broken' in elem]
            name = '%d_%s_%s.log' % (len(healthy_logs) + 1, self.runid, logname)

        filepath = os.path.join(log_folder, name)
        return filepath
