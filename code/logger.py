import random
import os
import glob
import datetime 
import shutil

class Logger():
    def __init__(self, args, time=False, depth=2, revision=False): 
        '''
        Initialisiation of logger class.
        * args: list of arguments (sys.argv) used to run the file from cmdline
        * time: boolean global option to log time for each logging entry
        * depth: int to determine folder depth and file naming convention (only relevant if revision=False)
        * revision: boolean to circumvent date/timestamp file and folder naming convention. 
        Calls for function to create directory based on time stamp, create the log file
        and copy the running file to directory. 
        '''
        self.args = args
        self.time = time
        self.depth = depth
        self.revision = revision
        self.prefix = '' # can be altered in self.create_directory()
        self.create_directory()
        self.initialise_logfile()
        self.backup_file()

    
    # --- class initialisation functions --- # 
    def create_directory(self):
        '''
        Creates a directory with timestamp in the the form of logs/date/time/filename
        and sets self.path (directory) and self.log_path (directory/log.log) as variables.
        '''
        fname = self.args[0].replace('.py', '')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date, time = timestamp.split(' ')
        dir_path = 'logs'
        
        # determine prefix and logging directory based on depth
        if not self.revision:
            if self.depth == 0: 
                self.prefix = timestamp.replace(' ', '_') + '_' + fname + '_' 
            elif self.depth == 1: 
                dir_path = os.path.join(dir_path, fname)
                self.prefix = timestamp.replace(' ', '_') + '_'
            elif self.depth == 2: 
                dir_path = os.path.join(dir_path, fname, timestamp)
            elif self.depth == 3: 
                dir_path = os.path.join(dir_path, fname, date, time)
            else: 
                raise Exception('Directory depth for logger is unclear.')
        else: 
            revs = glob.glob(os.path.join(dir_path, fname + '_*.log'))
            self.prefix = fname + '_' + str(len(revs)+1) + '_'
            
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.path = dir_path
        
        
    def initialise_logfile(self): 
        '''
        Creates the log file and writes parameters to it as first entry. 
        '''
        self.log_path = os.path.join(self.path, self.prefix + 'log.log')
        with open(self.log_path, 'w') as f: 
            f.write(self.format_params())
    
    def backup_file(self): 
        ''' 
        Writes a copy of the running file to the log folder.
        '''
        src = self.args[0]
        dst = os.path.join(self.path, self.prefix+src.split('/')[-1])
        shutil.copy(src, dst)
    
    
    # --- writing functions --- #            
    def write_to_file(self, text, code, time=False): 
        ''' 
        Generic write to file function. 
        * text: the string to be appended to the file.
        * code: ERROR, INFO, RESULT etc. 
        * time: extra parameter to ensure time is written along with the
                message, even if self.time is false.
        '''
        with open(self.log_path, 'a') as f: 
            f.write('\n' + code + str(text))
            if time or self.time: 
                f.write(self.get_time())
                
    def result(self, text, time=False): 
        ''' Writes result to log file. ''' 
        self.write_to_file(text, '[RESULT] \t', time)
    def info(self, text, time=False): 
        ''' Writes result to log file. ''' 
        self.write_to_file(text, '[INFO] \t \t', time)
    def error(self, text, time=False): 
        ''' Writes result to log file. ''' 
        self.write_to_file(text, '[ERROR] \t', time)
        
    # --- helper functions --- # 
    def get_time(self):
        ''' Returns timestamp in appropriate format. ''' 
        timestamp = datetime.datetime.now().strftime(" (%Y-%m-%d %H:%M:%S) ")
        return timestamp
        
    def format_params(self): 
        ''' Returns parameter settings in appropriate format.'''
        string = self.get_time() + 'Started ' + self.args[0] + ' with'
        if len(self.args) > 2: 
            for param in self.args[1:]:
                string += '\t' + param
        else: 
            string += ' no parameters.'
        return string 