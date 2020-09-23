from datetime import datetime

def timeit(method):
    def timed(obj, *args, **kw):
        ts = datetime.now()
        result = method(obj, *args, **kw)
        te = datetime.now()

        name = kw.get('log_name', method.__name__)
        if hasattr(obj, 'log'):
            obj.log[name] = te-ts
        if 'log' in kw:
            kw['log'][name] = te-ts
        if hasattr(obj, 'emit') and obj.emit:
            print('[{}] {} : {}'.format(datetime.now(), name, te-ts))

        return result

    return timed
