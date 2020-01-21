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

### Helper Functions

def is_date(string, fuzzy=True):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def find_dates(here):
  dict_date = {}
  for name in here.dtype.names:
    if here.dtype.fields[name][0] =="|U10":
      try:
        dict_date[name] = is_date(here[name][0])
      except:
        dict_date[name] = False
  return dict_date


def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b

## Array Functions

def drop(a, name_list):
  if (type(name_list)==str):
    name_list = [name_list]
  """
  `a` must be a numpy structured array.
  `names` is the collection of field names to remove.

  Returns a view of the array `a` (not a copy).
  """
  dt = a.dtype
  keep_names = [name for name in dt.names if name not in name_list]
  return view_fields(a, keep_names)

def add(array, name_list, value_list,types=None):
  if (len(name_list)==1) or (type(name_list)==str):
    if (types == None):
      try:
        types = value_list.dtype
      except:
        types = type(value_list)
        #return print("please specify type for single column")
    if (type(name_list)==str):
      name = name_list
    else:
      name = name_list[0]
    column_to_be_added = np.zeros(array.shape,dtype = [(name, types)])
    if type(value_list)=="list":
      column_to_be_added[name] = value_list[0]
    else:
      column_to_be_added[name] = value_list
    array = rfn.merge_arrays((array, column_to_be_added), asrecarray=False, flatten=True) 
  else:
    array = rfn.append_fields(array,name_list,value_list,usemask=False)
  
  return array

def update(array, column, values,types=None):
  if types==None:
    types= array.dtype.fields[column][0]
  array = drop(array,column)
  array = drop(array,column)
  array = add(array,column,values,types)
  return array


# %timeit (dfa["High"]/dfa["Low"])*dfa["Low"]
# %timeit (new["High"]/new["Low"])*new["Low"]

# tsla["Adj_Close"][:5]/100
# tsla["Adj_Close"][:5]*crm["Adj_Close"][:5]

def flip(array):
  return np.flip(array)


def rename(array,original, new):
  if (type(original)==str):
    original = [original]
    new = [new]
  mapping = {}
  for ori, ne in zip(original, new):
    mapping[ori] = ne
  
  return rfn.rename_fields(array,mapping)

## slower read function - work on it
def read(path):
  here = np.genfromtxt(path,delimiter=',', names=True, dtype=None, encoding=None,invalid_raise = False)
  dict_date = find_dates(here)
  for item in dict_date.keys():
    value = np.array([parse(d, fuzzy=False) for d in here[item]],dtype="M8[D]")
    here = drop(here, [item])
    here = add(here, item,value,"M8[D]")
  return here

def concat(first, second, type="row"):
  if type=="row":
    try:
      concat = np.concatenate([first, second])
    except:
      concat = np.concatenate([rfn.structured_to_unstructured(first), rfn.structured_to_unstructured(second)])
      concat = rfn.unstructured_to_structured(concat,names=first.dtype.names)
  if type=="columns":
    concat = rfn.merge_arrays((first, second), asrecarray=False, flatten=True) 
  if type=="array":
    concat = np.c_[[first, second]]
  if type=="melt": ## looks similar to columns but list instead of tuples
    try:
      concat = np.c_[(first, second)]
    except:
      concat = np.c_[(rfn.structured_to_unstructured(first), rfn.structured_to_unstructured(second))]
      concat = rfn.unstructured_to_structured(concat,names=first.dtype.names)
  return concat

def merge(left_array, right_array, left_on, right_on, how="inner", left_postscript="_left", right_postscript="_right" ):
  # DATA
  if left_on != right_on:
    if left_on in right_array.dtype.names:
      right_array = drop(right_array,left_on)

    mapping = {right_on: left_on}
    # LOGIC
    right_array.dtype.names = [mapping.get(word, word) for word in right_array.dtype.names]

  return rfn.join_by(left_on,left_array, right_array,jointype=how, usemask=False,r1postfix=left_postscript,r2postfix=right_postscript)

def replace(array, original=1.00000000e+020, replacement=np.nan):
  return np.where(array==1.00000000e+020, np.nan, array)

def columntype(array):
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
def ffill(arr):
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(1, out.shape[1]):
            if np.isnan(out[row_idx, col_idx]):
                out[row_idx, col_idx] = out[row_idx, col_idx - 1]
    return out

# My modification to do a backward-fill
def bfill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def fillmean(array):
  array = np.where(np.isnan(array), np.ma.array(array, 
                mask = np.isnan(array)).mean(axis = 0), array) 
  return array

def fillna(array, type="mean", value=None):

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


def table(array, length=None, row_values=None, column_values=None, value_name=None):
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

    if lenght != None:
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
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def pivot(array, row, column, value, display=True):
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
def group(array,groupby,compute,values,display=True, length=None):

  args_dict = {}
  for a in values:
    for f in compute:
      args_dict[a+"_"+f] = npg.aggregate(np.unique(array[groupby], return_inverse=True)[1], array[a],f)

  struct_gb = rfn.unstructured_to_structured(np.c_[list(args_dict.values())].T,names=list(args_dict.keys()))
  grouped = np.unique(array[groupby], return_inverse=True)[0]
  group_frame = rfn.merge_arrays([grouped,struct_gb],flatten=True)
  if display:
    table(group_frame,length)
  return group_frame

def pandas(array):

    is_unstructured = (array.dtype.names == None)
    if is_unstructured == True:
      raise ValueError("Arrays must have the same size")
    else:
      return pd.DataFrame(array)

#grouped_frame_two = grouped_frame.astype({name:str(grouped.dtype.fields[name][0]) for name in grouped.dtype.names})
def structured(pands):
  return pands.to_records(index=False)

#tsla_new_rem = lags(tsla_new_rem, "Adj_Close", 5)
def lags(array, feature, lags):
  for lag in range(1, lags + 1):
      col = '{}_lag_{}' .format(feature, lag)  
      array = add(array,col, shift(array[feature],lag), float)
  return array

#corr_mat = corr(closing)
def corr(array):
  corr_mat_price = np.corrcoef(rfn.structured_to_unstructured(array).T);
  table(corr_mat_price,array.dtype.names, column_values=array.dtype.names,value_name="Correlation")
  return corr_mat_price


def describe(array):
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
  table(fill_array.T,names,col_keys,"Describe")
  return fill_array

def ffill(array):
  mask = np.isnan(array)
  idx = np.where(~mask,np.arange(mask.shape[1]),0)
  np.maximum.accumulate(idx,axis=1, out=idx)
  array[mask] = array[np.nonzero(mask)[0], idx[mask]]
  return array

### outliers

### std_signal = (signal - np.mean(signal)) / np.std(signal)

@nb.jit
def detect(signal, treshold = 2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected
    
### Noise Filtering

@nb.jit
def removal(signal, repeat):
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal

### Get the noise
@nb.jit
def get(original_signal, removed_signal):
    buffer = []
    for i in range(len(removed_signal)):
        buffer.append(original_signal[i] - removed_signal[i])
    return np.array(buffer)



## ===================================================================================
## ===================================================================================

## ===================================================================================
## ===================================================================================

def returns(array, col, type):
  if type=="log":
    return np.log(array[col]/shift(array[col], 1))
  elif type=="normal":
    return array[col]/shift(array[col], 1) - 1

def portfolio_value(array, col, type):
  if type=="normal":
    return np.cumprod(array[col]+1) 
  if type=="log":
    return np.cumprod(np.exp(array[col]))


def cummulative_return(array, col, type):
  if type=="normal":
    return np.cumprod(array[col]+1) - 1
  if type=="log":
    return np.cumprod(np.exp(array[col])) - 1

def dropnarow(array, col):
  return array[~np.isnan(array[col])]

def subset(array, fields):
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
