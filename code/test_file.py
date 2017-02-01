import time
from control import ProgramEnder

ender = ProgramEnder()
while not ender.terminate:
        time.sleep(10)
        print('Hi.')

print 'Exiting gracefully.'

