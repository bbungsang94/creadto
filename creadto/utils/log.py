import os
import copy
import sys
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
import pandas as pd
from collections import defaultdict
from contextlib import contextmanager

from creadto.utils.io import make_dir


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


"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def write_dictionary(self, dictionary):
        raise NotImplementedError


class SeqWriter(object):
    def write_seq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def write_dictionary(self, dictionary):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(dictionary.items()):
            if hasattr(val, "__float__"):
                line = "%-8.3g" % val
            else:
                line = str(val)
            key2str[self._truncate(key)] = self._truncate(line)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (key_width - len(key)), val, " " * (val_width - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def write_seq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def write_dictionary(self, dictionary):
        for k, v in sorted(dictionary.items()):
            if hasattr(v, "dtype"):
                dictionary[k] = float(v)
        self.file.write(json.dumps(dictionary) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def write_dictionary(self, dictionary):
        # Add our current row to the history
        extra_keys = list(dictionary.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = dictionary.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()

def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def log_keyvalue(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().log_keyvalue(key, val)


def log_keyvalue_mean(key, val):
    """
    The same as log_keyvalue(), but if called many times, values averaged.
    """
    get_current().log_keyvalue_mean(key, val)


def log_dictionary(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        log_keyvalue(k, v)


def dump_dictionary():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dump_dictionary()


def get_dictionary():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    get_current().set_comm(comm)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = log_keyvalue
dump_tabular = dump_dictionary


@contextmanager
def profile_kv(scope_name):
    log_key = "wait_" + scope_name
    begin = time.time()
    try:
        yield
    finally:
        get_current().name2val[log_key] += time.time() - begin


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, directory, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = directory
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def log_keyvalue(self, key, val):
        self.name2val[key] = val

    def log_keyvalue_mean(self, key, val):
        old_val, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = old_val * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dump_dictionary(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.write_dict(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.write_seq(map(str, args))


def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    
    # MPI (Message Passing Interface)
    # PMI (Process Management Interface)
    # OMPI (OpenMPI)
    return 0


def mpi_weighted_mean(comm, local_name2tup):
    """
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2tup: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2tup = comm.gather(local_name2tup)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for name_tup in all_name2tup:
            for (name, (val, count)) in name_tup.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(directory=None, format_strs=None, comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if directory is None:
        directory = os.getenv("OPENAI_LOGDIR")
    if directory is None:
        directory = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(directory, str)
    directory = os.path.expanduser(directory)
    os.makedirs(os.path.expanduser(directory), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, directory, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(directory=directory, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % directory)


def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prev_logger = Logger.CURRENT
    configure(directory=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prev_logger