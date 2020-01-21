## PandaPy

A Structured NumPy Array is an array of structures. NumPy arrays can only contain one data type, but structured arrays in a sense create an array of homogeneous structures. This is done without moving out of NumPy such as is required with Xarray. For structured arrays the data type only has to be the same per column like an SQL data base. Each column can be another multidimensional object and does not have to conform to the basic NumPy datatypes.

Structured datatypes are designed to be able to mimic ‘structs’ in the C language, and share a similar memory layout. They are meant for interfacing with C code and for low-level manipulation of structured buffers, for example for interpreting binary blobs. For these purposes they support specialized features such as subarrays, nested datatypes, and unions, and allow control over the memory layout of the structure.

PandaPy comes with similar functionality like Pandas, such as groupby, pivot, and others. The biggest benefit of this approach is that NumPy dtype(data type) directly maps onto a C structure definition, so the buffer containing the array content can be accessed directly within an appropriately written C program. If you find yourself writing a Python interface to a legacy C or Fortran library that manipulates structured data, you'll probably find structured arrays quite useful. 
