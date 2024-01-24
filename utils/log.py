import copy
import os

import pandas as pd

from utils.io import make_dir


class CSVWriter:
    def __init__(self, path):
        self.save_folder = path
        make_dir(path)
        self.datum = dict()

    def add_scalars(self, column, data, **kwargs):
        data = copy.deepcopy(data)
        if column in self.datum:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        self.datum[column][key] += value
                    else:
                        self.datum[column][key].append(value)
            elif isinstance(data, list):
                self.datum[column] += data
            else:
                self.datum[column].append(data)
        else:
            if isinstance(data, dict):
                self.datum[column] = dict()
                for key, value in data.items():
                    self.datum[column][key] = [value]
            elif isinstance(data, list):
                self.datum[column] = data
            else:
                self.datum[column] = [data]

    def flush(self):
        datum = copy.deepcopy(self.datum)
        del_cols = []
        for key, value in datum.items():
            if isinstance(value, dict):
                del_cols.append(key)
                df = pd.DataFrame(data=value)
                df.to_csv(os.path.join(self.save_folder, key + '.csv'), index=False)
        for col in del_cols:
            datum.pop(col, None)

        if len(datum) > 0:
            df = pd.DataFrame(data=datum)
            df.to_csv(os.path.join(self.save_folder, 'Summary.csv'), index=False)


def print_message(message: str, width=60, line='-', center=False, padding=0):
    text = ''
    msg_len = len(message)
    line_len = width - msg_len - 2 * padding
    if line_len < 0:
        assert "Invalid width size(" + str(width) + "), message length is " + str(msg_len + 2 * padding)
    if center:
        text += line * (line_len // 2)
        text += ' ' * padding + message + ' ' * padding
        text += line * (line_len // 2)
        if line_len % 2 == 1:
            text += line
    else:
        text += ' ' * padding + message + ' ' * padding
        text += line * line_len
    return text
