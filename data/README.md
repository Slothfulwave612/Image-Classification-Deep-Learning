# Cat vs Non Cat Dataset

* The data directory contains the dataset in .h5 format.

## h5 files

* An H5 file is a data file saved in the Hierarchical Data Format (HDF). It contains multidimensional arrays of scientific data.

* Two commonly used versions of HDF include HDF4 and HDF5 (developed to improve upon limitations of the HDF4 library). Files saved in the HDF4 version are saved as an .H4 or HDF4 file. Files saved in the HDF5 version are saved as an H5 or HDF5 file.

## Handling h5 files(Python)

* **h5py** package is used.

* The h5py package is a Pythonic interface to the HDF5 binary data format. It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.

* To know more about h5py package, go to: https://www.geeksforgeeks.org/hdf5-files-in-python/

## About the dataset

* Our data directory includes two datasets:
  1. train_catvnoncat.h5: This h5 file contains our train dataset.
  2. test_catvnoncat.h5: This h5 file contains our test dataset.
  
* **train_catvnoncat.h5**:
  
  * A labelled dataset containing images and the label it is reffering to.
  * label *1* refers to *a cat* and label *0* refers to *non-cat*.
  * Example of our training instance:
  
    ![cat1](https://user-images.githubusercontent.com/33928040/74097234-c6dade00-4b2f-11ea-90fd-a9695f83a55c.PNG)   
    
    ![noncat2](https://user-images.githubusercontent.com/33928040/74097237-d1957300-4b2f-11ea-8efc-0c468724e39a.PNG)

* **test_catvnoncat.h5**:

  * It's our test dataset.
  * Example of out testing dataset:
  
    ![t1](https://user-images.githubusercontent.com/33928040/74097291-6d26e380-4b30-11ea-8000-320a3440359f.PNG)
    
    ![t2](https://user-images.githubusercontent.com/33928040/74097297-77e17880-4b30-11ea-9eee-c1c83ce1c338.PNG)

