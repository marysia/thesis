import numpy as np

def data_metrics(data, log):
    log.info('Metrics for %s dataset.' % data.name)
    data_types = [data.train, data.val, data.test]
    data_types = [dtype for dtype in data_types if 'empty' not in dtype.scope]
    for data_type in data_types:
        log.info('\nScope: %s \t shape x: %s \t shape y: %s' % (data_type.scope, str(data_type.x.shape), str(data_type.y.shape)))

        nb_classes = data_type.y.shape[1]
        total = np.sum(data_type.y)
        per_class = sum(data_type.y)
        msg = 'Class division: \t '
        for i in xrange(nb_classes):
            msg += '%d: %.1f%% \t' % (i, (per_class[i]/float(total))*100)
        log.info(msg)
