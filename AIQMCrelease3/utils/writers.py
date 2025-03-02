import contextlib
import os
from typing import Optional, Sequence
from absl import logging


class Writer(contextlib.AbstractContextManager):
    def __init__(self,
                 name: str,
                 schema: Sequence[str],
                 directory: str = 'logs/',
                 iteration_key: Optional[str] = 't',
                 log: bool = True):
        self._schema = schema
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self._filename = os.path.join(directory, name + '.csv')
        self._iteration_key = iteration_key
        self._log = log

    def __enter__(self):
        self._file = open(self._filename, 'w', encoding='UTF-8')
        if self._iteration_key:
            self._file.write(f'{self._iteration_key},')
        self._file.write(','.join(self._schema) + '\n')
        return self

    def write(self, t: int, ** data):
        row = [str(data.get(key, '')) for key in self._schema]
        if self._iteration_key:
            row.insert(0, str(t))
        for key in data:
            if key not in self._schema:
                raise ValueError(f'Not a recognized key for writer: {key}')
        self._file.write(','.join(row) + '\n')
        if self._log:
            logging.info('Iteration %s: %s', t, data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()