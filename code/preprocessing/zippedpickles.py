import pickle
import sys
import zipfile


class ZippedPickles:
    def __init__(self, file_name):
        # Open zip file.
        self.file = zipfile.ZipFile(file_name)

        self.names = [name[:-4] for name in self.file.namelist() if name.endswith('.pkl')]
        self.name_set = set(self.names)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getitem__(self, key):
        if key in self.name_set:
            key += '.pkl'
        else:
            raise Exception('Key \'{}\' was not found in archive.'.format(key))

        with self.file.open(key) as f:
            return pickle.load(f)

    def __contains__(self, key):
        return key in self.name_set

    def __iter__(self):
        return iter(self.names)

    def items(self):
        for name in self.names:
            yield (name, self[name])

    def keys(self):
        return iter(self.names)


# Since Python 3.6 we can write directly to a zip file.
if sys.version_info >= (3, 6):
    def _write(file_name, append, values):
        mode = 'a' if append else 'w'

        with zipfile.ZipFile(file_name, mode) as zip_file:
            # Dump all values to zip.
            for name, value in values.items():
                with zip_file.open('{}.pkl'.format(name), 'w', force_zip64=True) as f:
                    # Dump pickled value to file.
                    pickle.dump(value, f, protocol=4)
else:
    import tempfile
    import os


    def _write(file_name, append, values):
        mode = 'a' if append else 'w'
        directory = os.path.dirname(file_name)

        with zipfile.ZipFile(file_name, mode) as zip_file:
            for name, value in values.items():
                # Open temporary file in directory.
                with tempfile.NamedTemporaryFile(dir=directory) as temp_file:
                    # Dump pickled value to file.
                    pickle.dump(value, temp_file, protocol=4)

                    # Flush file to disk.
                    temp_file.flush()

                    # Add file to zip.
                    zip_file.write(temp_file.name, '{}.pkl'.format(name))


def save(file_name, **kwargs):
    _write(file_name, False, kwargs)


def append(file_name, **kwargs):
    _write(file_name, True, kwargs)


def load(file_name):
    return ZippedPickles(file_name)
