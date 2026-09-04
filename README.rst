Dataset suite (dataset-suite)
=============================

**Store scientific data in HDF5 files, but with pleasant
Python bindings!**

A general data handling toolkit with a powerful class for abstracting away work
with multidimensional set.

* Documentation: https://dataset-suite.readthedocs.io
* Repository: https://github.com/jdranczewski/dataset-suite-pip

Parts of this code were created as part of PhD work supported by the EU ITN EID
project CORAL (GA no. 859841).

Installation
------------
You can install this package from pip::

    pip install dataset-suite

Once installed, all useful functions are in the top-level module::

    import dataset_suite as ds


Basic usage
-----------
A short introduction to using the library::

    import dataset_suite as ds

    # Create a dataset out of a set of spectra
    data = ds.dataset(
        array,
        index=np.arange(array.shape[0]) # A basic numerical index
        wavelength=spectrometer.wavelength # You can store a numpy array as an axis descriptor
        # alongside your data. This way you know that the second axis of this dataset is a
        # "wavelength", and what wavelength corresponds to each datapoint
    )
    data.metadata["comment"] = "afternoon spectra"
    # Save the dataset to a HDF5 file
    data.save_h5("file.h5")
    # Load a dataset from a file
    data = ds.load_h5('file.h5')
    # inspect the dataset - what axes does it have?
    print(data)
    # dataset(index[10], wavelength[100])

    data.raw # access the underlying numpy array
    data.wl # use axis names as attributes to get the values for a given axis,
    #         like wavelength here
    data.wl[10] # this is the wavelength corresponding to the 10th column of the array
    data.take(index=5) # slice along the index axis: dataset(wavelength[100])
    data.take_sum("wavelength") # sum along the wavelength axis: dataset(index[10])
    data.metadata # dictionary of measurement metadata

The library also includes some convenient helpers like a function mapping values to
colours using matplotlib colormaps, and a number of filename search/analysis tools.
Have a look at https://dataset-suite.readthedocs.io/en/latest/api.html

