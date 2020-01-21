## PandaPy

A Structured NumPy Array is an array of structures. NumPy arrays can only contain one data type, but structured arrays in a sense create an array of homogeneous structures. This is done without moving out of NumPy such as is required with Xarray. For structured arrays the data type only has to be the same per column like an SQL data base. Each column can be another multidimensional object and does not have to conform to the basic NumPy datatypes.

Structured datatypes are designed to be able to mimic ‘structs’ in the C language, and share a similar memory layout. They are meant for interfacing with C code and for low-level manipulation of structured buffers, for example for interpreting binary blobs. For these purposes they support specialized features such as subarrays, nested datatypes, and unions, and allow control over the memory layout of the structure.

PandaPy comes with similar functionality like Pandas, such as groupby, pivot, and others. The biggest benefit of this approach is that NumPy dtype(data type) directly maps onto a C structure definition, so the buffer containing the array content can be accessed directly within an appropriately written C program. If you find yourself writing a Python interface to a legacy C or Fortran library that manipulates structured data, you'll probably find structured arrays quite useful. 



Getting observations just for the month of May


```python
!pip install numpy_groupies

```

    Collecting numpy_groupies
    [?25l  Downloading https://files.pythonhosted.org/packages/57/ae/18217b57ba3e4bb8a44ecbfc161ed065f6d1b90c75d404bd6ba8d6f024e2/numpy_groupies-0.9.10.tar.gz (43kB)
    [K     |████████████████████████████████| 51kB 2.0MB/s 
    [?25hBuilding wheels for collected packages: numpy-groupies
      Building wheel for numpy-groupies (setup.py) ... [?25l[?25hdone
      Created wheel for numpy-groupies: filename=numpy_groupies-0+unknown-cp36-none-any.whl size=28044 sha256=ea9465e6b060aca3c00c873713cf061739910e7edab233fb1d401b0ba69458d2
      Stored in directory: /root/.cache/pip/wheels/30/ac/83/64d5f9293aeaec63f9539142fc629a41af064cae1b3d8d94aa
    Successfully built numpy-groupies
    Installing collected packages: numpy-groupies
    Successfully installed numpy-groupies-0+unknown



```python
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

```

Read In Arrays


```python
paper = read('https://github.com/firmai/random-assets-two/raw/master/numpy/paper.csv')
tsla = read('https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv')
crm = read('https://github.com/firmai/random-assets-two/raw/master/numpy/crm.csv')
tsla_sub = tsla[["Date","Adj_Close","Volume"]]
crm_sub = crm[["Date","Adj_Close","Volume"]]
crm_adj = crm[['Date','Adj_Close']]


multiple_stocks = read('https://github.com/firmai/random-assets-two/blob/master/numpy/multiple_stocks.csv?raw=true')
closing = multiple_stocks[['Ticker','Date','Adj_Close']]
piv = pivot(closing,"Date","Ticker","Adj_Close"); piv
closing = rfn.unstructured_to_structured(piv, names=[x for x in np.unique(multiple_stocks["Ticker"])])
```


```python
closing
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312,  41.9791832 ,  81.51140594, 66.33999634),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312,  41.59314346,  80.89860535, 66.15000153),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897,  40.98268127,  80.28580475, 64.58000183),
           ...,
           (21.57999992, 289.79998779, 59.08000183, 11.18000031, 135.27000427, 55.34999847, 158.96000671, 137.53999329, 88.37000275),
           (21.34000015, 291.51998901, 58.65999985, 11.07999992, 132.80999756, 55.27000046, 157.58999634, 136.80999756, 87.95999908),
           (21.51000023, 293.6499939 , 58.47999954, 11.15999985, 134.03999329, 55.34999847, 157.69999695, 136.66999817, 88.08999634)],
          dtype=[('AA', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8')])




```python
rename(closing,["AA","AAPL"],["GAP","FAF"])[:5]
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939)],
          dtype=[('GAP', '<f8'), ('FAF', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8')])




```python
rename(closing,"AA", "GALLY")[:5]
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939)],
          dtype=[('GALLY', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8')])




```python
described = describe(closing)
```


<table><tr><th>Describe<th>observations<th>minimum<th>maximum<th>mean<th>variance<th>skewness<th>kurtosis<tr><th>AA<td>1258.00<td>15.97<td>60.23<td>31.46<td>99.42<td>0.67<td>-0.58<tr><th>AAPL<td>1258.00<td>85.39<td>293.65<td>149.45<td>2119.86<td>0.66<td>-0.28<tr><th>DAL<td>1258.00<td>30.73<td>62.69<td>47.15<td>44.33<td>-0.01<td>-0.78<tr><th>GE<td>1258.00<td>6.42<td>28.67<td>18.85<td>48.45<td>-0.25<td>-1.54<tr><th>IBM<td>1258.00<td>99.83<td>161.17<td>133.35<td>116.28<td>-0.37<td>0.56<tr><th>KO<td>1258.00<td>32.81<td>55.35<td>41.67<td>28.86<td>0.80<td>-0.05<tr><th>MSFT<td>1258.00<td>36.27<td>158.96<td>78.31<td>1102.21<td>0.61<td>-0.82<tr><th>PEP<td>1258.00<td>78.46<td>139.30<td>102.86<td>229.01<td>0.63<td>-0.32<tr><th>UAL<td>1258.00<td>37.75<td>96.70<td>69.22<td>195.65<td>0.02<td>-1.04</table>



```python
removed = drop(closing,["AA","AAPL","IBM"]) ; removed[:5]
```




    array([(44.57522202, 20.72605705, 35.80251312, 41.9791832 , 81.51140594, 66.33999634),
           (43.83200836, 20.34561157, 35.80251312, 41.59314346, 80.89860535, 66.15000153),
           (42.79874039, 19.90727234, 36.07437897, 40.98268127, 80.28580475, 64.58000183),
           (42.57216263, 19.91554451, 36.52467346, 41.50337982, 82.63342285, 65.52999878),
           (43.67792892, 20.15538216, 36.966465  , 42.72432327, 84.13523865, 66.63999939)],
          dtype={'names':['DAL','GE','KO','MSFT','PEP','UAL'], 'formats':['<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[16,24,40,48,56,64], 'itemsize':72})




```python
closing[:5]
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939)],
          dtype=[('AA', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8')])




```python
added = add(closing,["GALLY","FAF"],[closing["IBM"],closing["AA"]]); added[:5]  ## set two new columns with that two previous columnns
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634, 130.59109497, 37.24206924),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153, 128.53627014, 35.08446503),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183, 125.76422119, 35.34244537),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878, 124.94229126, 36.25707626),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939, 127.65791321, 37.28897095)],
          dtype=[('AA', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8'), ('GALLY', '<f8'), ('FAF', '<f8')])




```python
concat_row = concat(removed[["DAL","GE"]], added[["PEP","UAL"]], type="row"); concat_row[:5]
```




    array([(44.57522202, 20.72605705), (43.83200836, 20.34561157),
           (42.79874039, 19.90727234), (42.57216263, 19.91554451),
           (43.67792892, 20.15538216)], dtype=[('DAL', '<f8'), ('GE', '<f8')])




```python
concat_col = concat(removed[["DAL","GE"]], added[["PEP","UAL"]], type="columns"); concat_col[:5]
```




    array([(44.57522202, 20.72605705, 81.51140594, 66.33999634),
           (43.83200836, 20.34561157, 80.89860535, 66.15000153),
           (42.79874039, 19.90727234, 80.28580475, 64.58000183),
           (42.57216263, 19.91554451, 82.63342285, 65.52999878),
           (43.67792892, 20.15538216, 84.13523865, 66.63999939)],
          dtype=[('DAL', '<f8'), ('GE', '<f8'), ('PEP', '<f8'), ('UAL', '<f8')])




```python
concat_array = concat(removed[["DAL","GE"]], added[["PEP","UAL"]], type="array"); concat_array[:5]
```




    array([[(44.57522201538086, 20.726057052612305),
            (43.832008361816406, 20.345611572265625),
            (42.79874038696289, 19.907272338867188), ...,
            (59.08000183105469, 11.180000305175781),
            (58.65999984741211, 11.079999923706055),
            (58.47999954223633, 11.15999984741211)],
           [(81.51140594482422, 66.33999633789062),
            (80.89860534667969, 66.1500015258789),
            (80.28580474853516, 64.58000183105469), ...,
            (137.5399932861328, 88.37000274658203),
            (136.80999755859375, 87.95999908447266),
            (136.6699981689453, 88.08999633789062)]], dtype=object)




```python
concat_melt = concat(removed[["DAL","GE"]], added[["PEP","UAL"]], type="melt"); concat_melt[:5]
```




    array([(44.57522202, 20.72605705), (43.83200836, 20.34561157),
           (42.79874039, 19.90727234), (42.57216263, 19.91554451),
           (43.67792892, 20.15538216)], dtype=[('DAL', '<f8'), ('GE', '<f8')])




```python
merged = merge(tsla_sub, crm_adj, left_on="Date", right_on="Date",how="inner",left_postscript="_TSLA",right_postscript="_CRM"); merged[:5]
```




    array([('2019-01-02', 310.11999512, 135.55000305, 11658600),
           ('2019-01-03', 300.35998535, 130.3999939 ,  6965200),
           ('2019-01-04', 317.69000244, 137.96000671,  7394100),
           ('2019-01-07', 334.95999146, 142.22000122,  7551200),
           ('2019-01-08', 335.3500061 , 145.72000122,  7008500)],
          dtype=[('Date', '<M8[D]'), ('Adj_Close_TSLA', '<f8'), ('Adj_Close_CRM', '<f8'), ('Volume', '<i8')])




```python
## More work to done on replace (structured)
## replace(merged,original=317.69000244, replacement=np.nan)[:5]
```


```python
table(merged[:5])
```


<table><tr><th><th>Date<th>Adj_Close_TSLA<th>Adj_Close_CRM<th>Volume<tr><th>0<td>2019-01-02<td>310.120<td>135.550<td>11658600<tr><th>1<td>2019-01-03<td>300.360<td>130.400<td>6965200<tr><th>2<td>2019-01-04<td>317.690<td>137.960<td>7394100<tr><th>3<td>2019-01-07<td>334.960<td>142.220<td>7551200<tr><th>4<td>2019-01-08<td>335.350<td>145.720<td>7008500</table>



```python
### This is the new function that you should include above
### You can add the same peculuarities to remove
```


```python
tsla = add(tsla,["Ticker"], "TSLA", "U10")
crm = add(crm,["Ticker"], "CRM", "U10")
combine = np.concatenate([tsla[0:5], crm[0:5]]); combine
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, '2019-01-02', 'TSLA'),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, '2019-01-03', 'TSLA'),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, '2019-01-04', 'TSLA'),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, '2019-01-07', 'TSLA'),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , '2019-01-08', 'TSLA'),
           (136.83000183, 133.05000305, 133.3999939 , 135.55000305,  4783900, 135.55000305, '2019-01-02', 'CRM'),
           (134.77999878, 130.1000061 , 133.47999573, 130.3999939 ,  6365700, 130.3999939 , '2019-01-03', 'CRM'),
           (139.32000732, 132.22000122, 133.5       , 137.96000671,  6650600, 137.96000671, '2019-01-04', 'CRM'),
           (143.38999939, 138.78999329, 141.02000427, 142.22000122,  9064800, 142.22000122, '2019-01-07', 'CRM'),
           (146.46000671, 142.88999939, 144.72999573, 145.72000122,  9057300, 145.72000122, '2019-01-08', 'CRM')],
          dtype=[('High', '<f8'), ('Low', '<f8'), ('Open', '<f8'), ('Close', '<f8'), ('Volume', '<i8'), ('Adj_Close', '<f8'), ('Date', '<M8[D]'), ('Ticker', '<U10')])




```python
dropped = drop(combine,["High","Low","Open"]); dropped[:10]
```




    array([(310.11999512, 11658600, 310.11999512, '2019-01-02', 'TSLA'),
           (300.35998535,  6965200, 300.35998535, '2019-01-03', 'TSLA'),
           (317.69000244,  7394100, 317.69000244, '2019-01-04', 'TSLA'),
           (334.95999146,  7551200, 334.95999146, '2019-01-07', 'TSLA'),
           (335.3500061 ,  7008500, 335.3500061 , '2019-01-08', 'TSLA'),
           (135.55000305,  4783900, 135.55000305, '2019-01-02', 'CRM'),
           (130.3999939 ,  6365700, 130.3999939 , '2019-01-03', 'CRM'),
           (137.96000671,  6650600, 137.96000671, '2019-01-04', 'CRM'),
           (142.22000122,  9064800, 142.22000122, '2019-01-07', 'CRM'),
           (145.72000122,  9057300, 145.72000122, '2019-01-08', 'CRM')],
          dtype={'names':['Close','Volume','Adj_Close','Date','Ticker'], 'formats':['<f8','<i8','<f8','<M8[D]','<U10'], 'offsets':[24,32,40,48,56], 'itemsize':96})




```python
piv = pivot(dropped,"Date","Ticker","Adj_Close",display=True)
```


<table><tr><th>Adj_Close<th>CRM<th>TSLA<tr><th>2019-01-02<td>135.55<td>310.12<tr><th>2019-01-03<td>130.40<td>300.36<tr><th>2019-01-04<td>137.96<td>317.69<tr><th>2019-01-07<td>142.22<td>334.96<tr><th>2019-01-08<td>145.72<td>335.35</table>



```python
tsla_extended = add(tsla,"Month",tsla["Date"],'datetime64[M]')
tsla_extended = add(tsla_extended,"Year",tsla_extended["Date"],'datetime64[Y]')

```


```python
def update(array, column, values,types=None):
  if types==None:
    types= array.dtype.fields[column][0]
    print(types)
  array = drop(array,column)
  array = drop(array,column)
  array = add(array,column,values,types)
  return array
```


```python
## faster method elsewhere
year_frame = update(tsla,"Date", [dt.year for dt in tsla["Date"].astype(object)],types="|U10"); year_frame[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, 'TSLA', '2019'),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, 'TSLA', '2019'),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, 'TSLA', '2019'),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, 'TSLA', '2019'),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , 'TSLA', '2019')],
          dtype=[('High', '<f8'), ('Low', '<f8'), ('Open', '<f8'), ('Close', '<f8'), ('Volume', '<i8'), ('Adj_Close', '<f8'), ('Ticker', '<U10'), ('Date', '<U10')])




```python
grouped = group(tsla_extended, ['Ticker','Month','Year'],['mean', 'std', 'min', 'max'], ['Adj_Close','Close'], display=True)

```


<table><tr><th><th>Ticker<th>Month<th>Year<th>Adj_Close_mean<th>Adj_Close_std<th>Adj_Close_min<th>Adj_Close_max<th>Close_mean<th>Close_std<th>Close_min<th>Close_max<tr><th>0<td>TSLA<td>2019-01-01<td>2019-01-01<td>318.494<td>21.098<td>287.590<td>347.310<td>318.494<td>21.098<td>287.590<td>347.310<tr><th>1<td>TSLA<td>2019-02-01<td>2019-01-01<td>307.728<td>8.053<td>291.230<td>321.350<td>307.728<td>8.053<td>291.230<td>321.350<tr><th>2<td>TSLA<td>2019-03-01<td>2019-01-01<td>277.757<td>8.925<td>260.420<td>294.790<td>277.757<td>8.925<td>260.420<td>294.790<tr><th>3<td>TSLA<td>2019-04-01<td>2019-01-01<td>266.656<td>14.985<td>235.140<td>291.810<td>266.656<td>14.985<td>235.140<td>291.810<tr><th>4<td>TSLA<td>2019-05-01<td>2019-01-01<td>219.715<td>24.040<td>185.160<td>255.340<td>219.715<td>24.040<td>185.160<td>255.340<tr><th>5<td>TSLA<td>2019-06-01<td>2019-01-01<td>213.717<td>12.125<td>178.970<td>226.430<td>213.717<td>12.125<td>178.970<td>226.430<tr><th>6<td>TSLA<td>2019-07-01<td>2019-01-01<td>242.382<td>12.077<td>224.550<td>264.880<td>242.382<td>12.077<td>224.550<td>264.880<tr><th>7<td>TSLA<td>2019-08-01<td>2019-01-01<td>225.103<td>7.831<td>211.400<td>238.300<td>225.103<td>7.831<td>211.400<td>238.300<tr><th>8<td>TSLA<td>2019-09-01<td>2019-01-01<td>237.261<td>8.436<td>220.680<td>247.100<td>237.261<td>8.436<td>220.680<td>247.100<tr><th>9<td>TSLA<td>2019-10-01<td>2019-01-01<td>266.355<td>31.463<td>231.430<td>328.130<td>266.355<td>31.463<td>231.430<td>328.130<tr><th>10<td>TSLA<td>2019-11-01<td>2019-01-01<td>338.300<td>13.226<td>313.310<td>359.520<td>338.300<td>13.226<td>313.310<td>359.520<tr><th>11<td>TSLA<td>2019-12-01<td>2019-01-01<td>377.695<td>36.183<td>330.370<td>430.940<td>377.695<td>36.183<td>330.370<td>430.940</table>



```python
grouped_frame = pandas(grouped); grouped_frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Month</th>
      <th>Year</th>
      <th>Adj_Close_mean</th>
      <th>Adj_Close_std</th>
      <th>Adj_Close_min</th>
      <th>Adj_Close_max</th>
      <th>Close_mean</th>
      <th>Close_std</th>
      <th>Close_min</th>
      <th>Close_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TSLA</td>
      <td>2019-01-01</td>
      <td>2019-01-01</td>
      <td>318.494284</td>
      <td>21.098362</td>
      <td>287.589996</td>
      <td>347.309998</td>
      <td>318.494284</td>
      <td>21.098362</td>
      <td>287.589996</td>
      <td>347.309998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TSLA</td>
      <td>2019-02-01</td>
      <td>2019-01-01</td>
      <td>307.728421</td>
      <td>8.052522</td>
      <td>291.230011</td>
      <td>321.350006</td>
      <td>307.728421</td>
      <td>8.052522</td>
      <td>291.230011</td>
      <td>321.350006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TSLA</td>
      <td>2019-03-01</td>
      <td>2019-01-01</td>
      <td>277.757140</td>
      <td>8.924873</td>
      <td>260.420013</td>
      <td>294.790009</td>
      <td>277.757140</td>
      <td>8.924873</td>
      <td>260.420013</td>
      <td>294.790009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TSLA</td>
      <td>2019-04-01</td>
      <td>2019-01-01</td>
      <td>266.655716</td>
      <td>14.984572</td>
      <td>235.139999</td>
      <td>291.809998</td>
      <td>266.655716</td>
      <td>14.984572</td>
      <td>235.139999</td>
      <td>291.809998</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TSLA</td>
      <td>2019-05-01</td>
      <td>2019-01-01</td>
      <td>219.715454</td>
      <td>24.039647</td>
      <td>185.160004</td>
      <td>255.339996</td>
      <td>219.715454</td>
      <td>24.039647</td>
      <td>185.160004</td>
      <td>255.339996</td>
    </tr>
  </tbody>
</table>
</div>




```python
struct = structured(grouped_frame); struct[:5]
```




    rec.array([('TSLA', '2019-01-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', 318.49428449, 21.09836186, 287.58999634, 347.30999756, 318.49428449, 21.09836186, 287.58999634, 347.30999756),
               ('TSLA', '2019-02-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', 307.72842086,  8.05252198, 291.23001099, 321.3500061 , 307.72842086,  8.05252198, 291.23001099, 321.3500061 ),
               ('TSLA', '2019-03-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', 277.75713966,  8.92487345, 260.42001343, 294.79000854, 277.75713966,  8.92487345, 260.42001343, 294.79000854),
               ('TSLA', '2019-04-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', 266.65571594, 14.98457194, 235.13999939, 291.80999756, 266.65571594, 14.98457194, 235.13999939, 291.80999756),
               ('TSLA', '2019-05-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', 219.7154541 , 24.03964724, 185.16000366, 255.33999634, 219.7154541 , 24.03964724, 185.16000366, 255.33999634)],
              dtype=[('Ticker', 'O'), ('Month', '<M8[ns]'), ('Year', '<M8[ns]'), ('Adj_Close_mean', '<f8'), ('Adj_Close_std', '<f8'), ('Adj_Close_min', '<f8'), ('Adj_Close_max', '<f8'), ('Close_mean', '<f8'), ('Close_std', '<f8'), ('Close_min', '<f8'), ('Close_max', '<f8')])




```python
shift(merged["Adj_Close_TSLA"],1)[:5]
```




    array([         nan, 310.11999512, 300.35998535, 317.69000244,
           334.95999146])




```python
#tsla_new_rem = lags(tsla_new_rem, "Adj_Close", 5)
def lags(array, feature, lags):
  for lag in range(1, lags + 1):
      col = '{}_lag_{}' .format(feature, lag)  
      array = add(array,col,shift(array[feature],lag),float)
  return array
```


```python
tsla_lagged = lags(tsla_extended, "Adj_Close", 5); tsla_lagged[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, '2019-01-02', 'TSLA', '2019-01', '2019',          nan,          nan,          nan,          nan, nan),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, '2019-01-03', 'TSLA', '2019-01', '2019', 310.11999512,          nan,          nan,          nan, nan),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, '2019-01-04', 'TSLA', '2019-01', '2019', 300.35998535, 310.11999512,          nan,          nan, nan),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, '2019-01-07', 'TSLA', '2019-01', '2019', 317.69000244, 300.35998535, 310.11999512,          nan, nan),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , '2019-01-08', 'TSLA', '2019-01', '2019', 334.95999146, 317.69000244, 300.35998535, 310.11999512, nan)],
          dtype=[('High', '<f8'), ('Low', '<f8'), ('Open', '<f8'), ('Close', '<f8'), ('Volume', '<i8'), ('Adj_Close', '<f8'), ('Date', '<M8[D]'), ('Ticker', '<U10'), ('Month', '<M8[M]'), ('Year', '<M8[Y]'), ('Adj_Close_lag_1', '<f8'), ('Adj_Close_lag_2', '<f8'), ('Adj_Close_lag_3', '<f8'), ('Adj_Close_lag_4', '<f8'), ('Adj_Close_lag_5', '<f8')])




```python
correlated = corr(closing)
```


<table><tr><th>Correlation<th>AA<th>AAPL<th>DAL<th>GE<th>IBM<th>KO<th>MSFT<th>PEP<th>UAL<tr><th>AA<td>1.00<td>0.21<td>0.24<td>-0.17<td>0.39<td>-0.09<td>0.05<td>-0.04<td>0.12<tr><th>AAPL<td>0.21<td>1.00<td>0.86<td>-0.83<td>0.22<td>0.85<td>0.94<td>0.85<td>0.82<tr><th>DAL<td>0.24<td>0.86<td>1.00<td>-0.78<td>0.14<td>0.79<td>0.86<td>0.78<td>0.86<tr><th>GE<td>-0.17<td>-0.83<td>-0.78<td>1.00<td>0.06<td>-0.76<td>-0.86<td>-0.69<td>-0.76<tr><th>IBM<td>0.39<td>0.22<td>0.14<td>0.06<td>1.00<td>0.07<td>0.15<td>0.24<td>0.18<tr><th>KO<td>-0.09<td>0.85<td>0.79<td>-0.76<td>0.07<td>1.00<td>0.94<td>0.96<td>0.74<tr><th>MSFT<td>0.05<td>0.94<td>0.86<td>-0.86<td>0.15<td>0.94<td>1.00<td>0.93<td>0.83<tr><th>PEP<td>-0.04<td>0.85<td>0.78<td>-0.69<td>0.24<td>0.96<td>0.93<td>1.00<td>0.75<tr><th>UAL<td>0.12<td>0.82<td>0.86<td>-0.76<td>0.18<td>0.74<td>0.83<td>0.75<td>1.00</table>



```python
returns(closing,"IBM",type="log")
```




    array([        nan, -0.01585991, -0.02180223, ...,  0.0026649 ,
           -0.0183533 ,  0.0092187 ])




```python
loga = returns(closing,"IBM",type="normal"); loga
```




    array([        nan, -0.0157348 , -0.02156628, ...,  0.00266845,
           -0.0181859 ,  0.00926132])




```python
close_ret = add(closing,"IBM_log_return",loga); close_ret[:5]
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634,         nan),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153, -0.0157348 ),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183, -0.02156628),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878, -0.00653548),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939,  0.02173501)],
          dtype=[('AA', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8'), ('IBM_log_return', '<f8')])




```python
close_ret_na = dropnarow(close_ret, "IBM_log_return"); close_ret[:5]
```




    array([(37.24206924, 100.45429993, 44.57522202, 20.72605705, 130.59109497, 35.80251312, 41.9791832 , 81.51140594, 66.33999634,         nan),
           (35.08446503,  97.62433624, 43.83200836, 20.34561157, 128.53627014, 35.80251312, 41.59314346, 80.89860535, 66.15000153, -0.0157348 ),
           (35.34244537,  97.63354492, 42.79874039, 19.90727234, 125.76422119, 36.07437897, 40.98268127, 80.28580475, 64.58000183, -0.02156628),
           (36.25707626,  99.00255585, 42.57216263, 19.91554451, 124.94229126, 36.52467346, 41.50337982, 82.63342285, 65.52999878, -0.00653548),
           (37.28897095, 102.80648041, 43.67792892, 20.15538216, 127.65791321, 36.966465  , 42.72432327, 84.13523865, 66.63999939,  0.02173501)],
          dtype=[('AA', '<f8'), ('AAPL', '<f8'), ('DAL', '<f8'), ('GE', '<f8'), ('IBM', '<f8'), ('KO', '<f8'), ('MSFT', '<f8'), ('PEP', '<f8'), ('UAL', '<f8'), ('IBM_log_return', '<f8')])




```python
portfolio_value(close_ret_na, "IBM_log_return", type="log")
```




    array([0.98438834, 0.96338604, 0.95711037, ..., 1.15115429, 1.13040872,
           1.14092643])




```python
cummulative_return(close_ret_na, "IBM_log_return", type="log")
```




    array([-0.01561166, -0.03661396, -0.04288963, ...,  0.15115429,
            0.13040872,  0.14092643])




```python
fillna(tsla_lagged,type="mean")[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, 272.95330665, 272.38631982, 271.75180703, 271.10991915, 270.48587024),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, 310.11999512, 272.38631982, 271.75180703, 271.10991915, 270.48587024),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, 300.35998535, 310.11999512, 271.75180703, 271.10991915, 270.48587024),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, 317.69000244, 300.35998535, 310.11999512, 271.10991915, 270.48587024),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , 334.95999146, 317.69000244, 300.35998535, 310.11999512, 270.48587024)],
          dtype={'names':['High','Low','Open','Close','Volume','Adj_Close','Adj_Close_lag_1','Adj_Close_lag_2','Adj_Close_lag_3','Adj_Close_lag_4','Adj_Close_lag_5'], 'formats':['<f8','<f8','<f8','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[0,8,16,24,32,40,112,120,128,136,144], 'itemsize':152})




```python
fillna(tsla_lagged,type="value",value=-999999)[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, -9.99999000e+05, -9.99999000e+05, -9.99999000e+05, -9.99999000e+05, -999999.),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535,  3.10119995e+02, -9.99999000e+05, -9.99999000e+05, -9.99999000e+05, -999999.),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244,  3.00359985e+02,  3.10119995e+02, -9.99999000e+05, -9.99999000e+05, -999999.),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146,  3.17690002e+02,  3.00359985e+02,  3.10119995e+02, -9.99999000e+05, -999999.),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 ,  3.34959991e+02,  3.17690002e+02,  3.00359985e+02,  3.10119995e+02, -999999.)],
          dtype={'names':['High','Low','Open','Close','Volume','Adj_Close','Adj_Close_lag_1','Adj_Close_lag_2','Adj_Close_lag_3','Adj_Close_lag_4','Adj_Close_lag_5'], 'formats':['<f8','<f8','<f8','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[0,8,16,24,32,40,112,120,128,136,144], 'itemsize':152})




```python
fillna(tsla_lagged,type="ffill")[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512,          nan,          nan,          nan,          nan, nan),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, 310.11999512,          nan,          nan,          nan, nan),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, 300.35998535, 310.11999512,          nan,          nan, nan),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, 317.69000244, 300.35998535, 310.11999512,          nan, nan),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , 334.95999146, 317.69000244, 300.35998535, 310.11999512, nan)],
          dtype={'names':['High','Low','Open','Close','Volume','Adj_Close','Adj_Close_lag_1','Adj_Close_lag_2','Adj_Close_lag_3','Adj_Close_lag_4','Adj_Close_lag_5'], 'formats':['<f8','<f8','<f8','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[0,8,16,24,32,40,112,120,128,136,144], 'itemsize':152})




```python
fillna(tsla_lagged,type="bfill")[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, 300.35998535, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, 317.69000244, 300.35998535, 310.11999512, 310.11999512, 310.11999512),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , 334.95999146, 317.69000244, 300.35998535, 310.11999512, 310.11999512)],
          dtype={'names':['High','Low','Open','Close','Volume','Adj_Close','Adj_Close_lag_1','Adj_Close_lag_2','Adj_Close_lag_3','Adj_Close_lag_4','Adj_Close_lag_5'], 'formats':['<f8','<f8','<f8','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[0,8,16,24,32,40,112,120,128,136,144], 'itemsize':152})




```python
fillna(tsla_lagged,type="ffill")[:5]
```




    array([(315.13000488, 298.79998779, 306.1000061 , 310.11999512, 11658600, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (309.3999939 , 297.38000488, 307.        , 300.35998535,  6965200, 300.35998535, 310.11999512, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (318.        , 302.73001099, 306.        , 317.69000244,  7394100, 317.69000244, 300.35998535, 310.11999512, 310.11999512, 310.11999512, 310.11999512),
           (336.73999023, 317.75      , 321.72000122, 334.95999146,  7551200, 334.95999146, 317.69000244, 300.35998535, 310.11999512, 310.11999512, 310.11999512),
           (344.01000977, 327.01998901, 341.95999146, 335.3500061 ,  7008500, 335.3500061 , 334.95999146, 317.69000244, 300.35998535, 310.11999512, 310.11999512)],
          dtype={'names':['High','Low','Open','Close','Volume','Adj_Close','Adj_Close_lag_1','Adj_Close_lag_2','Adj_Close_lag_3','Adj_Close_lag_4','Adj_Close_lag_5'], 'formats':['<f8','<f8','<f8','<f8','<i8','<f8','<f8','<f8','<f8','<f8','<f8'], 'offsets':[0,8,16,24,32,40,112,120,128,136,144], 'itemsize':152})




```python
table(tsla_lagged,5)
```


<table><tr><th><th>High<th>Low<th>Open<th>Close<th>Volume<th>Adj_Close<th>Date<th>Ticker<th>Month<th>Year<th>Adj_Close_lag_1<th>Adj_Close_lag_2<th>Adj_Close_lag_3<th>Adj_Close_lag_4<th>Adj_Close_lag_5<tr><th>0<td>315.130<td>298.800<td>306.100<td>310.120<td>11658600<td>310.120<td>2019-01-02<td>TSLA<td>2019-01-01<td>2019-01-01<td>nan<td>nan<td>nan<td>nan<td>nan<tr><th>1<td>309.400<td>297.380<td>307.000<td>300.360<td>6965200<td>300.360<td>2019-01-03<td>TSLA<td>2019-01-01<td>2019-01-01<td>310.120<td>nan<td>nan<td>nan<td>nan<tr><th>2<td>318.000<td>302.730<td>306.000<td>317.690<td>7394100<td>317.690<td>2019-01-04<td>TSLA<td>2019-01-01<td>2019-01-01<td>300.360<td>310.120<td>nan<td>nan<td>nan<tr><th>3<td>336.740<td>317.750<td>321.720<td>334.960<td>7551200<td>334.960<td>2019-01-07<td>TSLA<td>2019-01-01<td>2019-01-01<td>317.690<td>300.360<td>310.120<td>nan<td>nan<tr><th>4<td>344.010<td>327.020<td>341.960<td>335.350<td>7008500<td>335.350<td>2019-01-08<td>TSLA<td>2019-01-01<td>2019-01-01<td>334.960<td>317.690<td>300.360<td>310.120<td>nan</table>



```python
std_signal = (signal - np.mean(signal)) / np.std(signal)
```


```python
signal = tsla_lagged["Volume"]
z_signal = (signal - np.mean(signal)) / np.std(signal)
```


```python
tsla_lagged = add(tsla_lagged,"z_signal_volume",z_signal)z_signal
```


```python
outliers = detect(tsla_lagged["z_signal_volume"]); outliers
```




    [12, 40, 42, 64, 78, 79, 84, 95, 97, 98, 107, 141, 205, 206, 207]




```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
plt.plot(np.arange(len(tsla_lagged["Volume"])), tsla_lagged["Volume"])
plt.plot(np.arange(len(tsla_lagged["Volume"])), tsla_lagged["Volume"], 'X', label='outliers',markevery=outliers, c='r')
plt.legend()
plt.show()
```


![png](PandaPy_Package_files/PandaPy_Package_49_0.png)



```python
price_signal = tsla_lagged["Close"]
removed_signal = removal(price_signal, 30)
noise = get(price_signal, removed_signal)
```


```python
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(removed_signal)
plt.title('timeseries without noise')
plt.subplot(2, 1, 2)
plt.plot(noise)
plt.title('noise timeseries')
plt.show()
```


![png](PandaPy_Package_files/PandaPy_Package_51_0.png)

