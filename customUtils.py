# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:23:39 2019

@author: kartik
"""

import numpy as np
import warnings
from daal.data_management import  (HomogenNumericTable,convertToHomogen_Float64,readOnly,BlockDescriptor_Float64, MergedNumericTable, \
                                   FileDataSource, DataSource, DataSourceIface, InputDataArchive, OutputDataArchive, \
                                   Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream)

'''
Arguments: Numeric table, *args = 'head' or 'tail'
Returns array of numeric table. 
If *args is 'head' returns top 5 rows
If *args is 'tail' returns last 5 rows
'''
def getArrayFromNT(nT,*args):
    doubleBlock = BlockDescriptor_Float64()
    if not args:
        firstRow = 0
        lastRow = nT.getNumberOfRows()
        firstCol = 0
        lastCol = nT.getNumberOfColumns()
    else:
        if args[0] == "head":
            firstRow = 0
            lastRow = 5
            firstCol = 0
            lastCol = 5
        if args[0] == "tail":
            firstRow = nT.getNumberOfRows() - 5
            lastRow = nT.getNumberOfRows()
            firstCol = 0
            lastCol = 5
    nT.getBlockOfRows(firstRow, lastRow, readOnly, doubleBlock)
    getArray = doubleBlock.getArray()
    return getArray