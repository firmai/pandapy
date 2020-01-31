import numpy as np
import numba as nb
import numpy.lib.recfunctions as rfn 
import numpy_groupies as npg
import scipy.sparse as sps
from scipy import stats
import pandas as pd
from dateutil.parser import parse
import datetime as dt
from operator import itemgetter
from itertools import groupby
from IPython.core.display import display, HTML
from html import escape
from typing import Union


### Helper Functions

def is_date(string: str, fuzzy: bool = True) -> bool:
    """
    Identifies whether the string is a date

    :param string: str, string to check for date.
    :param fuzzy: bool, ignore unknown tokens in string if True.
    :return bool: bool, whether the string can be interpreted as a date.
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def find_dates(array: np.ndarray) -> dict:
    """
    Finds all columns with 10 characters (e.g., 2000-00-00, blatjang23)
    and identify whether they are indeed dates using is_date(), load a
    boolean of all the answers into a python dictioary keyed with names. 

    :param array: np.ndarray(struct), array that contains the newly loaded dataset.
    :return dict_date: dict, returns a dictionary of all length 10 columns 
                       and boolean is_date answers.
    """
    dict_date = {}
    for name in array.dtype.names:
      if array.dtype.fields[name][0] =="|U10":
        try:
          dict_date[name] = is_date(array[name][0])
        except:
          dict_date[name] = False
    return dict_date


def view_fields(array: np.ndarray, names: list) -> np.ndarray:
    """
    A method to obtain a view of a structured array. 

    :param array: np.ndarray(structured), must be a numpy structured array.
    :param names: list, is the collection of field (or column) names to keep.
    :return b: np.ndarray(structured), returns a view of the array `array` (not a copy).
    """
    dt = array.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = array.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = array.view(newdt)
    return b

## Array Functions

def drop(array: np.ndarray, name_list: Union[list, str]) -> np.ndarray: 
    """
    A method to drop columns (fields) from a structured array

    :param array: np.ndarray(structured), must be a numpy structured array.
    :param name_list: list or str, is the collection of field/s (or column/s) names to keep.
    :return : np.ndarray(structured), returns a view of the array `array` (not a copy).
    """
    if (type(name_list)==str):
      name_list = [name_list]

    dt = array.dtype
    keep_names = [name for name in dt.names if name not in name_list]
    return view_fields(array, keep_names)

#@nb.jit
def array_load(array: np.ndarray, newobs: np.ndarray) -> np.ndarray:
    """
    Loop to load columns of a previous array into a new empty array

    :param array: np.ndarray(structured), must be a numpy structured array.
    :param newobs: np.ndarray(structured), must be a numpy structured array.
    :return newobs : np.ndarray(structured), returns an array with loaded columns/fields
    """
    for n in array.dtype.names:
      newobs[n]=array[n]
    return newobs


def add(array: np.ndarray, name_list: Union[list,str], value_list: Union[list,np.ndarray,str,int,float],types: str = None) -> np.ndarray:
    
    """
    Add additional data to the structured array

    :param array: np.ndarray(structured), array to which additional columns will be added.
    :param name_list: list or str, the name of the additional column/s to be added.
    :param value_list: list,np.ndarray,str,int,float, the value/s to be added to the column/s
    :param type: str, you can specify the type when only adding one column; 
                      when not provided it is deduced from the data. 
    :return newobs : np.ndarray(structured), returns the full array with the newly added fields.
    """   

    dt = [(val, key) for (val, key) in array.dtype.descr if val!='']
    if (len(name_list)==1) or (type(name_list)==str):
        if (types == None):
            try:
                types = value_list[0].dtype
            except:
                types = type(value_list)
        if (type(name_list)==str):
            name = name_list
        else:
            name = name_list[0]
        dt = dt + [(name, types)]
        newobs = np.empty(array.shape, dtype=dt)
        array_load(array, newobs)
        if type(value_list)==list:
            newobs[name]=value_list[0]
        else:
            newobs[name]=value_list
    else:
        try:
          dt = dt + [(new, val.dtype.descr[0][1]) for new, val in zip(name_list ,value_list )]
        except:
          if (types == None):
              try:
                ty = type(value_list[0])
              except:
                ty = type(value_list)
          else:
            ty = types

          dt = dt + [(new, ty) for new in name_list]

        newobs = np.empty(array.shape, dtype=dt)
        newobs = array_load(array, newobs)

        if type(value_list)!=list:
          for new in name_list:
              newobs[new]=value_list
        else:
            for new, val in zip(name_list ,value_list ):
              newobs[new]=val
    return newobs

## Actually slow rather use array[col] = values, unless you want to change dtype
def update(array: np.ndarray, column: str, values: Union[str, np.ndarray, str, int, float],types: str = None) -> np.ndarray:
    """
    A data update alternative from array[col] = values, that should only be used when
    you want to specify a specific datatype, otherwise the array[col] option is much faster

    :param array: np.ndarray(structured), array that houses the column to be updated.
    :param column: str, the name of the column to be added.
    :param values: list or np.ndarray, the value/s used to update the original 'column'
    :return array : np.ndarray(structured), the updated array
    """   
    if types==None:
      types= array.dtype.fields[column][0]
    array = drop(array,column)
    array = drop(array,column)
    array = add(array,column,values,types)
    return array


def flip(array: np.ndarray) -> np.ndarray:
    """
    Reverse the order of the data

    :param array: np.ndarray(structured), structured array to flip
    :return : np.ndarray(structured), flipped array
    """  
    return np.flip(array)


def rename(array: np.ndarray,original: Union[list, str], new: Union[list, str]) -> np.ndarray:
    """
    Rename the structured array.

    :param array: np.ndarray(structured), array that houses the columns/fields to be renamed.
    :param original: list or str, the original column/s to be renamed.
    :param new: list or str, the new column/s names.
    :return : np.ndarray(structured), the renamed structured array.
    """   

    if (type(original)==str):
      original = [original]
      new = [new]
    mapping = {}
    for ori, ne in zip(original, new):
      mapping[ori] = ne
    
    return rfn.rename_fields(array,mapping)

def to_struct(array: np.ndarray, name_list: list) -> np.ndarray:
    """
    Convert an unstructured (homogenous) array to a structured array. The data types
    are automatically picked up by looking at the data, using numpy's recfunctions.

    :param array: np.ndarray, unstructured array (i.e., normal numpy array)
    :param name_list: list or str, the names to be given to the columns to be
                                   given to the newly created structured array.
    :return : np.ndarray(structured), the newly converted structured array
    """   
    return rfn.unstructured_to_structured(array, names= name_list)
  
def to_unstruct(array: np.ndarray) -> np.ndarray:
    """
    Convert an structured (non-homogenous) array to an unstructured array. 

    :param array: np.ndarray(structured), structured array (i.e., numpy array with columns/fields)
    :return : np.ndarray, the newly converted unstructured (normal) array
    """   
    return rfn.structured_to_unstructured(array)


def read(path: str, delimiter=",",convert_date=True) -> np.ndarray:
    """
    Read in dataframe from csv, url or other datatypes accepted by numpy. Identify
    which columns are dates and convert them to a date format, return structured array.  

    :param path: str, the path of the file to open and process.
    :param delimiter: str, what is the separator to use for the data.
    :return array : np.ndarray, loaded and processed structured array.
    """   
    array = np.genfromtxt(path,delimiter=delimiter, names=True, dtype=None, encoding=None,invalid_raise = False)
    if convert_date:
        dict_date = find_dates(array)
        for item in dict_date.keys():
            try:  
                value = array[item].astype("M8[D]")
            except:
                print("slow date conversion in progress")
                value = np.array([parse(d, fuzzy=False) for d in array[item]],dtype="M8[D]")
            array = drop(array, [item])
            array = add(array, item,value,"M8[D]")
        return array
    else:
        return array

def concat_col(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    A unique method that concatenates two structured numpy arrays
    by column and a bit of tidying up to remove void datatypes. A 
    new array is created with the new data types of the concatenating
    array included. The concatenating arrays are loaded into the empty
    array via a loop function array_load(). 

    Note, if you are concatenating a single column, always add double
    brackets so that the name can be easily retrieved i.e. array[[col]]

    :param array1: np.ndarray, the left concatenating array.
    :param array2: np.ndarray, the right concatenating array.
    :return newobs : np.ndarray, newly concatenated array.
    """   
    dt = [(val, key) for (val, key) in array1.dtype.descr if val!='']
    dt = dt +  [(val, key) for (val, key) in array2.dtype.descr if val!='']

    newobs = np.empty(array1.shape, dtype=dt)
    try:
        newobs = array_load(array1, newobs)
    except:
        print("Put additional brackets array1[[col]] instead of array1[col] OR use add() and not concatenate")

    try:
        newobs = array_load(array2, newobs)
    except:
        print("Put additional brackets array2[[col]] instead of array1[col] OR use add() and not concatenate")

    return newobs

def concat(first: np.ndarray, second: np.ndarray, type: "{row, columns, array, melt}" = "row") -> np.ndarray:
    """
    Multiple methods of concatenation, some of them are experimental. The basic methods are 'columns' or 'row'. 
    The other methods do not necessarily provide unique outcomes to that of 'columns' and 'row'.

    Note, if you are concatenating a single column, always add double
    brackets so that the name can be easily retrieved i.e. array[[col]]

    :param array1: np.ndarray, the left/top concatenating array.
    :param array2: np.ndarray, the right/bottom concatenating array.
    :param type: str or in, the type of concatenation 'row', 'columns', 'array' or 'melt'
    :return concat : np.ndarray, newly concatenated array.
    """
    if type in ["row","r","rows",0]:
      try:
        concat = np.concatenate([first, second])
      except:
        concat = np.concatenate([rfn.structured_to_unstructured(first), rfn.structured_to_unstructured(second)])
        concat = rfn.unstructured_to_structured(concat,names=first.dtype.names)
    elif type in ["columns","column","c",1]:
      concat = concat_col(first,second)
      #concat = rfn.merge_arrays((first, second), asrecarray=False, flatten=True)  # tuples
    elif type=="array":
      concat = np.c_[[first, second]]
    elif type=="melt": ## looks similar to columns but list instead of tuples
      try:
        concat = np.c_[(first, second)]
      except:
        concat = np.c_[(rfn.structured_to_unstructured(first), rfn.structured_to_unstructured(second))]
        concat = rfn.unstructured_to_structured(concat,names=first.dtype.names)
    else:
      raise ValueError("type has to be set to either: row, columns, array or melt")
    return concat

def merge(left_array: np.ndarray, right_array: np.ndarray, left_on: str, right_on: str, how: "{inner, outer, leftouter}" = "inner", left_postscript="_left", right_postscript="_right" ) -> np.ndarray:
    """
    Multiple methods of merging data on unique columns. This method is not optimised and makes use of NumPy's recfunctions. 
    This method achieves everything that can be done with Pandas' merge fucntion.

    :param left_array: np.ndarray, the left concatenating array.
    :param right_array: np.ndarray, the right concatenating array.
    :param left_on: str, the left unique column to merge on.
    :param right_on: str, the right unique column to merge on.
    :param how: {inner, outer, leftouter} str, 
        If 'inner', returns the elements common to both r1 and r2.
        If 'outer', returns the common elements as well as the elements of
        r1 not in r2 and the elements of not in r2.
        If 'leftouter', returns the common elements and the elements of r1
        not in r2.
    :param left_postscript: str, appended to the names of the fields of left_array that are present
        in right_array but absent of the key.
    :param right_postscript: str, appended to the names of the fields of right_array that are present
        in left_array but absent of the key.
    :return : np.ndarray, newly merged array.
    """

    # DATA
    if how not in ["inner","outer","leftouter"]:
      raise ValueError("how has to be set to either: 'inner','outer','leftouter'")
    if left_on != right_on:
      if left_on in right_array.dtype.names:
        right_array = drop(right_array,left_on)

      mapping = {right_on: left_on}
      # LOGIC
      right_array.dtype.names = [mapping.get(word, word) for word in right_array.dtype.names]

    return rfn.join_by(left_on,left_array, right_array,jointype=how, usemask=False,r1postfix=left_postscript,r2postfix=right_postscript)

def replace(array: np.ndarray, original=1.00000000e+020, replacement=np.nan) -> np.ndarray:
    """
    More to be done
    """
    return np.where(array==1.00000000e+020, np.nan, array)

def columntype(array: np.ndarray) -> tuple:
    """
    Take in an array and output columns in numeric and 
    non-numeric types. 

    :param array: np.ndarray, input array. 
    :return : (numeric_cols, nonnumeric_cols) a tuple of lists.
    """
    numeric_cols = []
    nonnumeric_cols = []
    for col in array.dtype.names:
      if (array[col].dtype == float) or (array[col].dtype == int):
        numeric_cols.append(col)
      else:
        nonnumeric_cols.append(col)
    return numeric_cols, nonnumeric_cols
    
# def bifill(array):
#   mask = np.isnan(array)
#   idx = np.where(~mask,np.arange(mask.shape[1]),0)
#   np.maximum.accumulate(idx,axis=1, out=idx)
#   array[mask] = array[np.nonzero(mask)[0], idx[mask]]
#   return array

@nb.njit
def ffill(arr: np.ndarray) -> np.ndarray:
    """
    Function to forward fill NaN values using Numba.  

    :param array: np.ndarray, input array with NaNs. 
    :return: np.ndarray, return forward filled array.
    """
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(1, out.shape[1]):
            if np.isnan(out[row_idx, col_idx]):
                out[row_idx, col_idx] = out[row_idx, col_idx - 1]
    return out

# My modification to do a backward-fill
def bfill(arr: np.ndarray) -> np.ndarray:
    """
    Function to backward fill NaN values.  

    :param array: np.ndarray, input array with NaNs. 
    :return: np.ndarray, return backward filled array.
    """

    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def fillmean(array) -> np.ndarray:
    """
    Function to backward fill NaN values with column mean.  

    :param array: np.ndarray, input array with NaNs. 
    :return: np.ndarray(structured), return mean filled array.
    """
    array = np.where(np.isnan(array), np.ma.array(array, 
                  mask = np.isnan(array)).mean(axis = 0), array) 
    return array

def fillna(array: np.ndarray, type="mean", value=None) -> np.ndarray:
    """
    List of filling functions applied to unstructured (normal) arrays and converted
    back to structured arrays as output.

    :param array: np.ndarray, input array with NaNs. 
    :param type: {mean, value, ffill, bfill} str,
        'mean' returns the column mean.
        'value' returns the value parameter.
        'ffill' returns a forward filled array.
        'bfill' returns a backwards filled array.
    :param value: optional, value to be used in type='value'.
    :param array: np.ndarray, input array with NaNs. 
    :return: np.ndarray, return mean filled array.
    """
    numeric_cols, nonnumeric_cols = columntype(array)
    dtyped = array[numeric_cols].dtype
    numeric_unstructured = rfn.structured_to_unstructured(array[numeric_cols]).T

    if type=="mean":
      numeric_unstructured = fillmean(numeric_unstructured.T)

    if type=="value":
      if value==None:
        value = 0
        print("To replace with anything different to 0, supply value=x")
      
      numeric_unstructured = np.nan_to_num(numeric_unstructured,nan= value)

    if type=="ffill":
      numeric_unstructured = ffill(numeric_unstructured)

    if type=="bfill": ## ffi
      numeric_unstructured = bfill(numeric_unstructured)

    if type=="mean":
      numeric_structured = rfn.unstructured_to_structured(numeric_unstructured,dtype=dtyped)
    else:
      numeric_structured = rfn.unstructured_to_structured(numeric_unstructured.T,dtype=dtyped)

    if len(array[nonnumeric_cols].dtype):
      full_return = numeric_structured
    else:
      full_return = concat(array[nonnumeric_cols],numeric_structured,type="columns")

    return full_return


def table(array: np.ndarray, length: int = None, row_values: list = None, column_values: list = None, value_name: str = None) -> None:
    """
    An HTML table to be printed of structured numpy arrays. 

    :param array: np.ndarray, input array to be printed. 
    :param length: int, how many rows to print
    :param row_values: list, a list of the structured array's row names.
    :param column_values: list,  a list of the structured array's column names.
    :param value_name: str, the descriptor to appear on the top left of the table. 
    :return: None print
    """

    if not row_values:
      row_values = range(len(array))
    if not column_values:
      column_values=array.dtype.names
    if not value_name:
      value_name=""

    fields_original = array.dtype.fields
    
    is_unstructured = (array.dtype.names == None)

    if is_unstructured == False:
      array = np.array(array,dtype='object')

    """Numpy array HTML representation function."""
    # Fallbacks for cases where we cannot format HTML tables
    if array.size > 10_000:
        return f"Large numpy array {array.shape} of {array.dtype}"
    if (array.ndim != 2) and (is_unstructured) :
        return f"<pre>{escape(str(array))}</pre>"

    # Table format
    html = [f"<table><tr><th>{value_name}"]
    html += (f"<th>{j}" for j in column_values)

    if length != None:
      row_values = row_values[:length]

    for i, rv in enumerate(row_values):
        html.append(f"<tr><th>{rv}")
        for j, cv in enumerate(column_values):
          if is_unstructured:
            val = array[i,j]
            html.append("<td>")
            html.append(escape(f"{val:.2f}" if array.dtype == float else f"{val}"))
          else:
            val = array[i][j]
            html.append("<td>")
            html.append(escape(f"{val:.3f}" if fields_original[cv][0] == "float" else f"{val}"))
    html.append("</table>")
    display(HTML("".join(html)))


# preallocate empty array and assign slice
def shift(array: np.ndarray, steps: int, fill_value=np.nan) -> np.ndarray:
    """
    Shift an array num number of steps forward. 

    :param array: np.ndarray, input array to be printed. 
    :param steps: int, how steps to shift.
    :param fill_value: optional, fill the values with anything other than NaN
    :return result: np.ndarray, return shifted array. 
    """
    result = np.empty_like(array)
    if steps > 0:
        result[:steps] = fill_value
        result[steps:] = array[:-steps]
    elif steps < 0:
        result[steps:] = fill_value
        result[:steps] = array[-steps:]
    else:
        result[:] = array
    return result

def pivot(array: np.ndarray, row: str, column: str, value: str, display: bool = True) -> np.ndarray:
    """
    Shift an array num number of steps forward. 

    :param array: np.ndarray, input array to be pivoted. 
    :param index: string, column to use to make new arrays's index. 
    :param columns: string, column to use to make new arrays's columns.
    :param values: string, column to use for populating new arrays's values.
    :param display: bool, whether or not to display a printed HTML data frame. 
    :return pivot_table: np.ndarray, pivoted array. 
    """

    rows, row_pos = np.unique(array[row], return_inverse=True)
    cols, col_pos = np.unique(array[column], return_inverse=True)

    pivot_table = np.zeros((len(rows), len(cols)), dtype=array.dtype)
    # pivot_table[row_pos, col_pos] = array["Adj_Close"]

    pivot_table = sps.coo_matrix((array[value], (row_pos, col_pos)),
                                shape=(len(rows), len(cols))).A

    if display:
      table(pivot_table,None, list(rows), list(cols), value)
    

    return pivot_table

# grouped = group(array, ['Ticker','Month','Year'],['mean', 'std', 'min', 'max'], ['Adj_Close','Close'],display=True)
# %timeit df.groupby(['Ticker','Month','Year'])[['Adj_Close','Close']].agg(['mean', 'std', 'min', 'max'])
# %timeit groupby(array, ['Ticker','Month','Year'],['mean', 'std', 'min', 'max'], ['Adj_Close','Close'], display=False)

# npg.aggregate(np.unique(tsla_extended[['Ticker','Month','Year']], return_inverse=True)[1], tsla_extended, func='last', fill_value=0)
def group(array: np.ndarray,groupby_cols: list, compute_functions: list, calcs_cols: list,display=True, length=None) -> np.ndarray:
    """
    Group the array according to a unique mapper of multiple columns (groupby_cols) by doing various calculations (compute_functions)
    over a select columns (calc_cols).

    :param array: np.ndarray, input array to be grouped.  
    :param groupby_cols: list, columns to be used to do the grouping.  
    :param compute_functions: list, columns to be used to specify the different calculations. 
    :param calcs_cols: list, columns over which the computations will be done. 
    :param display: bool, whether or not to display a printed HTML data frame. 
    :param length: int, how many rows of the displayed HTML table to print. 
    :return group_array: np.ndarray, grouped array. 
    """

    args_dict = {}
    for a in calcs_cols:
      for f in compute_functions:
        args_dict[a+"_"+f] = npg.aggregate(np.unique(array[groupby_cols], return_inverse=True)[1], array[a],f)

    struct_gb = rfn.unstructured_to_structured(np.c_[list(args_dict.values())].T,names=list(args_dict.keys()))
    grouped = np.unique(array[groupby_cols], return_inverse=True)[0]
    group_array = rfn.merge_arrays([grouped,struct_gb],flatten=True)
    if display:
      table(group_array,length)
    return group_array

def pandas(array: np.ndarray) -> pd.DataFrame:
    """
    Convert structured numpy to pandas dataframe. 

    :param array: np.ndarray, input array to be transformed into a pandas df.  
    :return : pd.DataFrame: pandas dataframe.
    """

    is_unstructured = (array.dtype.names == None)
    if is_unstructured == True:
      raise ValueError("Arrays must have the same size")
    else:
      return pd.DataFrame(array)

#grouped_frame_two = grouped_frame.astype({name:str(grouped.dtype.fields[name][0]) for name in grouped.dtype.names})
def structured(pands: pd.DataFrame) -> np.ndarray:
    """
    Convert pandas dataframe to structured numpy array. 

    :param pands: pd.DataFrame, pandas df to be transformed into structured numpy array.   
    :return : np.ndarray, structured array.
    """
    return pands.to_records(index=False)

#tsla_new_rem = lags(tsla_new_rem, "Adj_Close", 5)
def lags(array: np.ndarray, feature: str, lags: int) -> np.ndarray:
    """
    Create a range of lags for a certain feature (column), append and
    return the full array. 

    :param array: np.ndarray, array from which to calculate lagged columns.
    :param feature: str, name of column to be lagged. 
    :param lags: int, how many lag columns to be created with steps of one.   
    :return : np.ndarray, structured array with appended lags.
    """
    for lag in range(1, lags + 1):
        col = '{}_lag_{}' .format(feature, lag)  
        array = add(array,col, shift(array[feature],lag), float)
    return array

#corr_mat = corr(closing)
def corr(array: np.ndarray, display=True) -> np.ndarray:
    """
    Correlation matrix to be returned from an homogenous structured array.

    :param array: np.ndarray, array from which to derive correlation matrix.
    :param feature: str, name of column to be lagged. 
    :param lags: int, how many lag columns to be created with steps of one.   
    :param display: bool, whether or not to display a printed HTML data frame.
    :return : np.ndarray, correlation matrix in the format of an unstructured array.
    """
    corr_mat_price = np.corrcoef(rfn.structured_to_unstructured(array).T)
    if display!=False:
        table(corr_mat_price, None, array.dtype.names, column_values=array.dtype.names,value_name="Correlation")
    return corr_mat_price


def describe(array: np.ndarray, display=True) -> np.ndarray:
    """
    Descriptive statistics to be returned from a numerical array. 

    :param array: np.ndarray, array from which to derive descriptive statistics. 
    :param display: bool, whether or not to display a printed HTML data frame.
    :return : np.ndarray, descriptive statistics in the format of an unstructured array.
    """
    fill_array = np.zeros(shape=(7,len(array.dtype.names)))
    col_keys = ["observations", "minimum", "maximum", "mean", "variance", "skewness", "kurtosis"]
    en_dec = 0 
    names = []
    for en, arr in enumerate(array.dtype.names): #do not need the loop at this point, but looks prettier
      
      en = en - en_dec
      try:
        desc = stats.describe(array[arr])
        names.append(arr)
        
      except:
        fill_array = np.delete(fill_array,en,1)
        en_dec = en_dec + 1
        continue
      col_values = [desc[0], desc[1][0], desc[1][1], desc[2], desc[3], desc[4], desc[5]]
      # newrow = [1,2,3]
      # A = numpy.vstack([A, newrow])
      fill_array[:,en] = col_values
    fill_array = np.round(fill_array,3)
    if display==True:
      table(fill_array.T, None, names,col_keys,"Describe")
    return fill_array

### outliers

### std_signal = (signal - np.mean(signal)) / np.std(signal)

@nb.jit
def detect(signal: np.ndarray, treshold = 2.0) -> list:
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected
    
### Noise Filtering

@nb.jit
def removal(signal: np.ndarray, repeat: int) -> np.ndarray:
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal

### Get the noise
@nb.jit
def get(original_signal: np.ndarray, removed_signal: np.ndarray) -> np.ndarray:
    buffer = []
    for i in range(len(removed_signal)):
        buffer.append(original_signal[i] - removed_signal[i])
    return np.array(buffer)



## ===================================================================================
## ===================================================================================

## ===================================================================================
## ===================================================================================

def returns(array: np.ndarray, col: str, type: str) -> np.ndarray:
    """
    Singular column array of returns to be returned from a nominated column of the full array.
    The returns can be returned both in log and normal format.

    :param array: np.ndarray, array which houses the price column from which to calculate the returns. 
    :param col: str, the column name of the price series to be used in return calculation. 
    :param type: {'log', 'normal'} str, the type of return calculation to be returned. 
    :return : np.ndarray, the return series in singular array format. 
    """
    if type=="log":
      return np.log(array[col]/shift(array[col], 1))
    elif type=="normal":
      return array[col]/shift(array[col], 1) - 1

def portfolio_value(array, col, type) -> np.ndarray:
    """
    Singular column array of portfolio values to be returned from a nominated column of the full array.
    The portfolio value can be calculated from both a log and normal input format.

    :param array: np.ndarray, array which houses the returns column from which to calculate the returns. 
    :param col: str, the column name of the returns series to be used in portfolio value calculation. 
    :param type: {'log', 'normal'} str, the type of calculation to be ingested in the portfolio value calculation. 
    :return : np.ndarray, the portfolio value series in singular array format. 
    """
    if type=="normal":
      return np.cumprod(array[col]+1) 
    if type=="log":
      return np.cumprod(np.exp(array[col]))


def cummulative_return(array, col, type) -> np.ndarray:
    """
    Singular column array of cummulative returns to be returned from a nominated column of the full array.
    The cummulative returns can be calculated from both a log and normal input format.

    :param array: np.ndarray, array which houses the returns column from which to calculate the returns. 
    :param col: str, the column name of the returns series to be used in the cummulative returns calculation. 
    :param type: {'log', 'normal'} str, the type of calculation to be ingested in the cummulative returns calculation. 
    :return : np.ndarray, the cummulative returns series in singular array format. 
    """

    if type=="normal":
      return np.cumprod(array[col]+1) - 1
    if type=="log":
      return np.cumprod(np.exp(array[col])) - 1

def dropnarow(array, col) -> np.ndarray:
    """
    Dropping rows in a structured array where NaN values appear. 

    :param array: np.ndarray, which houses the array with NaN value rows.
    :param col: str, the column to look at when identifying the NaN value rows.  
    :return : np.ndarray, return the array without any NaN in the rows. 
    """
    return array[~np.isnan(array[col])]

def subset(array, fields) -> np.ndarray:
    """
    More to be done
    """
    
    return array.getfield(np.dtype(
        {name: array.dtype.fields[name] for name in fields}
    ))


## ===================================================================================
## ===================================================================================

## ===================================================================================
## ===================================================================================

# PMA = moving_average(combined_trends["debtP"], 3)
# OMA = moving_average(combined_trends["debtO"], 3)

def moving_average(a, n=5):
    a = np.ma.masked_array(a,np.isnan(a))
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan

    return ret

def moving_average(array,column, period):
    signal = array[column]
    buffer = [np.nan] * period
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer

def auto_regressive(array,column, p, d, q, future_count = 10):
    """
    p = the order (number of time lags)
    d = degree of differencing
    q = the order of the moving-average
    """
    signal = array[column]
    buffer = np.copy(signal).tolist()
    for i in range(future_count):
        ma = moving_average(np.array(buffer[-p:]), q)
        forecast = buffer[-1]
        for n in range(0, len(ma), d):
            forecast -= buffer[-1 - n] - ma[n]
        buffer.append(forecast)
    return buffer

# future_count = 15
# predicted_15 = auto_regressive(signal,15,1,2,future_count)
# predicted_30 = auto_regressive(signal,30,1,2,future_count)

def linear_weight_moving_average(array,column, period):
    signal = array[column]
    buffer = [np.nan] * period
    for i in range(period, len(signal)):
        buffer.append(
            (signal[i - period : i] * (np.arange(period) + 1)).sum()
            / (np.arange(period) + 1).sum()
        )
    return buffer

def anchor(array,column, weight):
    signal = array[column]
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
