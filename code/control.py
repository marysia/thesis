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
