"""
A general data handling toolkit with a powerful class for abstracting away work
with multidimensional set. Store scientific data in HDF5 files, but with pleasant
Python bindings!

Code by Jakub Dranczewski

MIT License

Copyright (c) 2026 Jakub Dranczewski
Parts of this code were created as part of PhD work supported by the EU ITN EID
project CORAL (GA no. 859841).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import abc
import numpy as np
import os
import re
import pickle
import gzip
from glob import glob as _glob
import h5py


def load(filename):
    """
    Load a dataset-suite object saved as a pickle with :func:`dataset_suite.base_dataobject.save`.
    """
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def load_h5(filename):
    """
    Load a dataset-suite object saved as a HDF5 file with :func:`dataset_suite.base_dataobject.save_h5`.
    """
    with h5py.File(filename, "r") as h5:
        return _from_h5_router(h5)


def _dict_to_h5(
    h5: h5py.File | h5py.Group, name: str, dictionary: dict, compression: int
) -> None:
    group: h5py.Group = h5.create_group(name, track_order=True)
    for key in dictionary:
        _to_h5_router(group, key, dictionary[key], compression)


def _h5_to_dict(h5: h5py.Group) -> dict:
    out = {}
    for key in h5:
        data = h5[key][...]
        if len(data.shape) == 0:
            data = data[()]
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        out[key] = data
    return out


def _to_h5_router(
    group: h5py.File | h5py.Group, key: str, data, compression: int
) -> h5py.Group | h5py.Dataset:
    if hasattr(data, "_populate_h5"):
        sub_group: h5py.Group = group.create_group(key, track_order=True)
        data._populate_h5(sub_group, compression)
        return sub_group
    if data is None:
        # Special case, handled as an Empty dataset
        sub_group: h5py.Dataset = group.create_dataset(
            key, dtype="int8"
        )
        return sub_group
    else:
        try:
            # Is it a numpy array, or something that can be cast to a homogeneous numpy array?
            sub_group: h5py.Dataset = group.create_dataset(
                key, data=data, compression="gzip", compression_opts=compression
            )
            return sub_group
        except (TypeError, ValueError):
            # Is it a scalar that can be stored as an un-compressed dataset?
            try:
                # Special case for strings
                # https://docs.h5py.org/en/latest/strings.html
                if isinstance(data, np.ndarray) and "U" in str(data.dtype):
                    data = str(data)
                sub_group: h5py.Dataset = group.create_dataset(
                    key,
                    data=data,
                )
                return sub_group
            except (TypeError, ValueError):
                # Is it a list or tuple?
                try:
                    iterator = enumerate(data)
                except TypeError:
                    raise TypeError(
                        f"Couldn't store data in an h5 file at {group} / {key}"
                    )
                sub_group: h5py.Group = group.create_group(key, track_order=True)
                sub_group.attrs["dataset_type"] = "list"
                for i, sub_data in iterator:
                    _to_h5_router(sub_group, str(i), sub_data, compression)
                return sub_group


def _from_h5_router(h5: h5py.File | h5py.Group | h5py.Dataset):
    if isinstance(h5, h5py.Dataset):
        value = h5[...]
        if value.shape is None:
            return None
        if not len(value.shape):
            # just a single value rather than an array
            value = value[()]
        return value
    if "dataset_type" not in h5.attrs:
        raise ValueError("This h5 object does not conform to dataset-suite standards.")
    if h5.attrs["dataset_type"] == "dataset":
        return dataset.from_h5(h5)
    elif h5.attrs["dataset_type"] == "datalist":
        return datalist.from_h5(h5)
    elif h5.attrs["dataset_type"] == "datadict":
        return datadict.from_h5(h5)
    elif h5.attrs["dataset_type"] == "list":
        return [_from_h5_router(h5[key]) for key in h5.keys()]
    raise ValueError(f"No valid dataset-suite object type found for {h5}")


class base_dataobject:
    def save(self, filename, compress=6) -> None:
        """
        A dataset, datadict, or datalist can be saved as a compressed file to the drive.
        Use :func:`dataset_suite.load` to get an object from said file.

        This method saves as a gzipped pickle, which is very flexible in terms of what objects
        and data can be saved, **but requires the user to have the libraries for all pickled
        objects installed**. If you pickle a datadict that includes a pytorch tensor, a person
        loading the file has to have pytorch installed. Mismatched package versions can also
        lead to trouble (most notably numpy <2.0 and >2.0).

        Generally it is advisable to use :func:`dataset_suite.base_dataobject.save_h5` and :func:`dataset_suite.load_h5` instead.
        """
        with gzip.open(filename, "wb", compresslevel=compress) as f:
            pickle.dump(self, f)

    def save_h5(self, filename: str, compress: int = 6) -> None:
        """
        A dataset, datadict, or datalist can be saved as a compressed HDF5 file to the drive.
        Use :func:`dataset_suite.load_h5` to get an object from said file.

        HDF5 files are a standard format, and you can use viewers like
        https://myhdf5.hdfgroup.org/ to inspect them.
        """
        with h5py.File(filename, "w", track_order=True) as f:
            self._populate_h5(f, compress)
            # It's possible that the _populate call above didn't create a metadata group, check and create
            # so we can write the filename into it
            meta_group = (
                f["metadata"]
                if "metadata" in f.keys()
                else f.create_group("metadata", track_order=True)
            )
            meta_group.create_dataset("filename", data=os.path.basename(filename))

    @classmethod
    def from_h5(cls, h5: h5py.File | h5py.Group):
        raise NotImplementedError

    def _populate_h5(self, h5: h5py.File | h5py.Group, compression: int) -> None:
        raise NotImplementedError


class dataset(base_dataobject):
    """
    A container for data arrays with labelled axes.

    Provide an array of data, and keyword arguments defining the values of the dataset's axes::

        ds.dataset(spectrum, pixel=np.arange(spectrum.shape[0]), wavelength=wavelengths)

    This way the axes of the data array are labelled, and information about what each row/column/...
    corresponds to is stored right with the data.
    """

    def __init__(self, data, cut=None, **axes):
        self._raw = np.asarray(data)
        if len(axes) != self._raw.ndim:
            raise IndexError("The number of provided axes does not match the dataset.")
        for i, key in enumerate(axes):
            if self._raw.shape[i] != len(axes[key]):
                raise IndexError(
                    "The shape of the provided axes does not match the dataset."
                )
            setattr(self, key, np.asarray(axes[key]))
        self._axes = list(axes.keys())
        if cut is None:
            cut = {}
        self._cut = cut
        self.metadata = {}

    def take(self, **i):
        """
        Return a slice of the dataset at the specified index for a given axis (as a dataset).

        For example::

            data.take(power=2)
        """
        s_raw = self._raw
        cut = self._cut
        new_axes = self.ax_dict
        new_ax_names = self.axes.copy()
        for key in i:
            s_raw = np.moveaxis(s_raw, new_ax_names.index(key), 0)[i[key]]
            cut[key] = self.axis(key)[i[key]]
            new_axes.pop(key)
            new_ax_names.remove(key)
        new_data = dataset(s_raw, cut=cut, **new_axes)
        if hasattr(self, "metadata"):
            new_data.metadata = self.metadata
        return new_data

    def take_raw(self, **i):
        """
        Return a slice of the dataset at the specified index for a given axis (as a numpy array).

        For example::

            data.take_raw(power=2)
            # equivalent to:
            data.take(power=2).raw
        """
        s_raw = self._raw
        new_ax_names = self.axes.copy()
        for key in i:
            s_raw = np.moveaxis(s_raw, new_ax_names.index(key), 0)[i[key]]
            new_ax_names.remove(key)
        return s_raw

    def take_sum(self, axis):
        """
        Sum this dataset along the named axis and return the resulting dataset.

        For example::

            data.take_sum("wavelength")
        """
        new_axes = self.ax_dict
        new_axes.pop(axis)
        new_data = dataset(np.sum(self._raw, axis=self._axes.index(axis)), **new_axes)
        if hasattr(self, "metadata"):
            new_data.metadata = self.metadata
        return new_data

    def expand(self, new_axis, value):
        return dataset(
            np.expand_dims(self._raw, axis=0), **{new_axis: [value], **self.ax_dict}
        )

    def join(self, other, axis):
        new_axes = self.ax_dict
        new_axes[axis] = np.concatenate((self.axis(axis), other.axis(axis)))
        return dataset(
            np.concatenate((self._raw, other.raw), axis=self._axes.index(axis)),
            **new_axes,
        )

    def axis(self, ax):
        return getattr(self, ax)

    def astype(self, _type):
        new_data = dataset(self._raw.astype(_type), cut=self._cut, **self.ax_dict)
        new_data.metadata = self.metadata
        return new_data

    @property
    def raw(self):
        return self._raw

    @property
    def axes(self):
        return self._axes

    @property
    def ax_dict(self):
        return {key: self.axis(key) for key in self._axes}

    @property
    def cut(self):
        return self._cut

    def add_cut(self, key, value):
        self._cut[key] = value

    def _populate_h5(self, h5: h5py.File | h5py.Group, compression: int) -> None:
        h5.attrs["dataset_type"] = "dataset"

        data_group: h5py.Group = h5.create_group("data", track_order=True)
        data_group.create_dataset(
            "values", data=self.raw, compression="gzip", compression_opts=compression
        )
        data_group.attrs["NX_class"] = "NXdata"
        data_group.attrs["signal"] = "values"

        # Add the axes to the main data group
        axes = []
        for key in self.ax_dict:
            data_group.create_dataset(
                key,
                data=self.ax_dict[key],
                compression="gzip",
                compression_opts=compression,
            )
            axes.append(key)
        data_group.attrs["axes"] = axes

        # Store metadata
        if len(self.metadata.keys()):
            _dict_to_h5(h5, "metadata", self.metadata, compression)
        if len(self.cut.keys()):
            _dict_to_h5(h5, "cut", self.cut, compression)

    @classmethod
    def from_h5(cls, h5: h5py.File | h5py.Group):
        axes = {
            axis_key: h5["data"][axis_key][...] for axis_key in h5["data"].attrs["axes"]
        }
        cut = _h5_to_dict(h5["cut"]) if "cut" in h5 else None
        obj = cls(h5["data"]["values"], cut, **axes)
        if "metadata" in h5:
            obj.metadata = _h5_to_dict(h5["metadata"])
        return obj

    def __repr__(self):
        return "dataset({})".format(
            ", ".join("{}[{}]".format(key, len(self.axis(key))) for key in self._axes)
        )


class datalist(base_dataobject):
    """
    A container for data, like a list, but with an axis array that can contain values.

    For example, this could store a dataset for each stage position in a scan (but generally
    you should consider if using a dataset directly may not be better).

    ``axis`` is the name for the datalists one axis.
    """

    def __init__(self, axis, cut=None):
        self._axes = [
            axis,
        ]
        self._axis = []
        self._datasets = []
        setattr(self, axis, self._axis)
        if cut is None:
            cut = {}
        self._cut = cut
        self.metadata = {}

    def append(self, ds, value):
        try:
            ds.add_cut(self._axes[0], value)
        except AttributeError:
            pass
        self._datasets.append(ds)
        self._axis.append(value)

    def add_cut(self, key, value):
        self._cut[key] = value
        for ds in self:
            try:
                ds.add_cut(key, value)
            except AttributeError:
                pass

    @property
    def axis(self):
        return self._axis

    @property
    def axes(self):
        return self._axes

    @property
    def datasets(self):
        return self._datasets

    @property
    def cut(self):
        return self._cut

    def _populate_h5(self, h5: h5py.File | h5py.Group, compression: int) -> None:
        h5.attrs["dataset_type"] = "datalist"

        data_group: h5py.Group = h5.create_group("data", track_order=True)
        data_group.attrs["axes"] = self.axes
        _to_h5_router(data_group, self.axes[0], self.axis, compression)
        for i, value, data in zip(range(len(self.axis)), self.axis, self._datasets):
            group = _to_h5_router(data_group, str(i), data, compression)
            group.attrs["axis_value"] = value

        # Store metadata
        if len(self.metadata.keys()):
            _dict_to_h5(h5, "metadata", self.metadata, compression)
        if len(self.cut.keys()):
            _dict_to_h5(h5, "cut", self.cut, compression)

    @classmethod
    def from_h5(cls, h5: h5py.File | h5py.Group):
        axis = h5["data"].attrs["axes"][0]
        cut = _h5_to_dict(h5["cut"]) if "cut" in h5 else None
        obj = cls(axis, cut)
        if "metadata" in h5:
            obj.metadata = _h5_to_dict(h5["metadata"])
        axis_values = h5["data"][axis][...]
        for i, axis_value in enumerate(axis_values):
            obj.append(_from_h5_router(h5["data"][str(i)]), axis_value)
        return obj

    def __getitem__(self, i: int):
        return self._datasets[i]

    def __len__(self):
        return len(self._datasets)

    def __repr__(self):
        return str(self._datasets)


class datadict(base_dataobject):
    """
    A key: value store for data like a Python dictionary, but with the metadata, cut, and save mechanics
    from the dataset object.

    ``name`` is the name of this datadict.
    """

    def __init__(self, name, cut=None):
        self._name = name
        self._dict = {}
        if cut is None:
            cut = {}
        self._cut = cut
        self.metadata = {}

    def add_cut(self, key, value):
        self._cut[key] = value
        for element in self:
            try:
                self._dict[element].add_cut(key, value)
            except AttributeError:
                pass

    @property
    def name(self):
        return self._name

    @property
    def dict(self):
        return self._dict

    @property
    def cut(self):
        return self._cut

    def __setitem__(self, key, value):
        try:
            value.add_cut(self._name, key)
        except AttributeError:
            pass

        self._dict[key] = value

    def __iter__(self):
        for key in self._dict:
            yield key

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, item):
        return item in self._dict

    def keys(self):
        return self._dict.keys()

    def _populate_h5(self, h5: h5py.File | h5py.Group, compression: int) -> None:
        h5.attrs["dataset_type"] = "datadict"
        h5.attrs["datadict_name"] = self.name

        data_group: h5py.Group = h5.create_group("data", track_order=True)
        for key in self:
            data = self[key]
            _to_h5_router(data_group, key, data, compression)

        # Store metadata
        if len(self.metadata.keys()):
            _dict_to_h5(h5, "metadata", self.metadata, compression)
        if len(self.cut.keys()):
            _dict_to_h5(h5, "cut", self.cut, compression)

    @classmethod
    def from_h5(cls, h5: h5py.File | h5py.Group):
        cut = _h5_to_dict(h5["cut"]) if "cut" in h5 else None
        obj = cls(h5.attrs["datadict_name"], cut)
        if "metadata" in h5:
            obj.metadata = _h5_to_dict(h5["metadata"])
        for key in h5["data"]:
            obj[key] = _from_h5_router(h5["data"][key])
        return obj

    def __repr__(self):
        return "datadict({}: {})".format(self._name, ", ".join(self._dict.keys()))


## Dictionary utilities


def print_dict(data: dict, offset=0):
    """
    Print a nicely formatted view of the data in a nested
    dictionary/datadict.
    """
    for key in data:
        print(f"{offset*4*' '}{key}:  ", end="")
        value = data[key]
        print_value(value, offset)
        print("")


def _handle_dict_value(key, value):
    """
    Convert value to a standar format - for use in
    :func:`dataset_suite.to_datadict`.
    """
    if isinstance(value, datadict):
        # Save datadicts directly
        return value
    elif isinstance(value, dict):
        # Convert dicts to datadicts recursively
        return to_datadict(key, value)
    else:
        # Attempt to convert other stuff to a numpy array
        try:
            array = np.asarray(value)
            if len(array.shape) == 0:
                # The value is a scalar
                return value
            if array.dtype == np.dtype("O"):
                raise ValueError("The value was a list of objects")
            return array
        except ValueError:
            if isinstance(value, abc.Iterable):
                # If the numpy array conversion fails, but the value is iterable, convert to a list of arrays
                output = datalist("list")
                for i, subvalue in enumerate(value):
                    output.append(_handle_dict_value(str(i), subvalue), i)
                return output
            else:
                # Give up
                return value


def to_datadict(name, data):
    """
    Convert a dictionary-like object to a datadict for saving.
    Tries its best to make numpy arrays out of things.
    """
    dd = datadict(name)
    for key in data:
        value = data[key]
        value = _handle_dict_value(key, value)
        # Save the converted (or not) value to the new datadict
        dd[key] = value
    return dd


## File traversal utilities


def glob(path, yes=[], no=[]):
    """
    A small extension to the excellent glob library. 'path' is defined as for the usual
    glob: a Unix-style pathname with * as a wildcard.
    'yes' and 'no' are lists/tuples or string that should/should not be in the full file
    name, so act as filters.
    """
    files = _glob(path)
    for s in yes:
        files = [x for x in files if s in x]
    for s in no:
        files = [x for x in files if s not in x]
    files.sort()
    return files


def folders(folder="", rel_current=True):
    """
    Returns a list of folders within a specified directory. Current directory by default.
    Set rel_current to False to return just the folder names, instead of the full path
    relative to current directory.
    """
    if rel_current:
        return [
            os.path.join(*list(os.path.split(x)[:-1]))
            for x in glob(os.path.join(folder, "*", ""))
        ]
    else:
        return [os.path.split(x)[-2] for x in glob(os.path.join(folder, "*", ""))]


def prefixes(folder, split, yes=[], no=[]):
    """
    Returns a list of file prefixes in a given folder that match some conditions.
    'prefix' is defined as the part of the filename before the string given as 'split'.
    'yes' and 'no' are lists/tuples or string that should/should not be in the full file
    name, so act as filters.
    """
    files = os.listdir(folder)
    for s in yes + [split]:
        files = [x for x in files if s in x]
    for s in no:
        files = [x for x in files if s not in x]
    prefixes = list(set([x.split(split)[0] for x in files]))
    prefixes.sort()
    return prefixes


def extract(string, *patterns):
    """
    Given a string and an arbitrary number of patterns as arguments, this function extracts
    associated numbers from the string. For example, '_P_' as a pattern will match '_P_123'
    in a filename and return 123.
    """
    return [
        int(re.search(pattern + "([0-9]+)", string).group(1)) for pattern in patterns
    ]


def extract_raw(string, *patterns):
    """
    You can write regex? Good for you.
    Given a string and an arbitrary number of patterns as arguments, this function extracts
    associated things from the string. You have to include a capture group, otherwise errors ensue.
    """
    return [re.search(pattern, string).group(1) for pattern in patterns]


def extract_unique(files, pattern):
    """
    Like :func:`dataset-suite.extract`, but returns a list of unique matched values in a list of filenames.
    """
    numbers = list(set([extract(x, pattern)[0] for x in files]))
    numbers.sort()
    return numbers


def _sort_key(x, pattern):
    try:
        return int(re.search(pattern + "([0-9]+)", x).group(1))
    except AttributeError:
        return 1e10


def sort_by(files, pattern):
    """
    Given a list of strings (like file names or sth) and a pattern, the functions extracts
    an associated numeric parameter from each string and sorts the list according to that
    parameter.

    For example, if your files contain power information as '_P_123', pass '_P_' as 'pattern'.
    Sorting happens in-place in the passed list, so nothing is returned.
    """
    files.sort(key=lambda x: _sort_key(x, pattern))


def colours(values, cmap: str | None = None, minmax=None):
    """
    Return a list of colours that a matplotlib colourmap would assign to the given
    values. ``minmax`` is a tuple of ``v_min, v_max`` boundaries for the colourmap.
    """
    # Check if matplotlib needs to be imported.
    # We only do this here, as this takes a bit of time, so it's silly to do this
    # every time the full library is imported.
    if "matplotlib" not in globals():
        global matplotlib
        import matplotlib

    # Default colourmap
    if cmap is None:
        cmap = "viridis"
    cmap_obj = matplotlib.colormaps[cmap]

    if minmax is None:
        a, b = np.amin(values), np.amax(values)
    else:
        a, b = minmax

    if len(values) == 1:
        return cmap_obj((0,))
    return cmap_obj((values - a) / (b - a))


def map_axes(data, **axes):
    """
    Helper function for figuring out how the axes you have map to the data.
    data is the ndarray, and then you pass your axes in any order as keyword arguments.
    """
    data = np.asarray(data)
    shape = data.shape
    for s in shape:
        print(s, [key for key in axes if len(axes[key]) == s])


def print_value(value, offset=0):
    """
    Print a value with standard formatting. Used in :func:`dataset_suite.print_dict`.
    """
    if isinstance(value, np.ndarray):
        if len(value.shape):
            print(f"array({value.shape})", end="")
        else:
            print(value, end="")
    elif isinstance(value, datadict) or isinstance(value, dict):
        print("")
        print_dict(value, offset + 1)
    elif isinstance(value, tuple) or isinstance(value, list):
        print("(", end="")
        for subvalue in value:
            print_value(subvalue, offset)
            print(", ", end="")
        print(")", end="")
    else:
        rep = str(value)
        if len(rep) > 60:
            rep = f"{rep[:29]}..{rep[-29:]}"
        print(rep, end="")
