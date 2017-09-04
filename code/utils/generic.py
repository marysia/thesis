import datetime
import time


def log_time(log, args, models):
    start_time = time.time() + (2 * 3600)  # system time is two hours behind; adjust.
    start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    log.info('Starting training at: \t %s' % start_str)

    if args.mode == 'time':
        end_time = start_time + ((args.mode_param * 60) * len(models.keys()))
        end_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Estimated end time: \t %s' % end_str)
