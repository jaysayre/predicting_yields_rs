import pickle
import numpy as np
import pandas as pd
import re
import os

raw_path = '/home/ubuntu/yield_data/'
processed_path = '/home/ubuntu/hists/'

# Fix codes so they can be matched to the shapefile
def fixEnt(code):
    stringCode = str(code)
    if len(stringCode)==1:
        stringCode = "0"+stringCode
    return stringCode

def fixMun(code):
    stringCode = str(code)
    if len(stringCode) == 2:
        stringCode = "0"+stringCode
    elif len(stringCode) == 1:
        stringCode = "00"+stringCode
    return stringCode

allYields = pd.read_pickle(raw_path+'allYields.pkl')

bands = list(map(lambda x:'b0'+str(x),list(range(1,8))))
bands.append('LST')

# Function to clean a given histogram cell
def cleanHistCell(cell):
    # Make a list of strings of format 'binstart, count'
    stringList = cell.replace('[[','[').replace(']]',']').strip('][').split('], [')
    # Make each element into a list of format [binstart, count]
    histList = list(map(lambda x: x.split(', '),stringList))
    # Get the list of count values for the histogram
    countList = [float(i[1]) for i in histList]
    # Normalize 'em
    countList = list(map(lambda x: x/sum(countList),countList))
    # return it
    return(countList)

# Function to make the input data
def makeInput(row):
    output = []
    # bin x time x band
    for i in range(0,32):
        if i == 0:
            suffix = ''
        else:
            suffix = '_'+str(i)

        bandsToAdd = list(map(lambda x: x+suffix,bands))
        rowBands = row[bandsToAdd]

        # Outputs an 8 element list (bnads) of 32 element lists (bins)
        binXband =rowBands.apply(lambda z: cleanHistCell(z)).tolist()
        output.append(binXband)
    # At this point output is in time x band x bin
    output = np.array(output)
    # rearrange as bin x time x band
    output = np.moveaxis(output,-1,0)
    output = output.tolist()
    return output

# Function to clean clean bad rows
def deleteBadHists(df):
    histCols = []
    for i in range(0,32):
        if i == 0:
            suffix = ''
        else:
            suffix = '_'+str(i)
        for band in bands:
            histCols.append(band+suffix)
    histCols.append('CVE_ENT')
    histCols.append('CVE_MUN')
    probs = [i for i in histCols if not i in list(df.columns)]
    if len(probs) == 0:
        dfOut = df[histCols]
        dfOut = dfOut.dropna(axis=0)
        return dfOut
    else:
        return None

# Function to take a given state and year and combine yields w/ the corr. csv
def combineStateYear(state, year):
    fileName = processed_path+'muni_hists_state_'+str(state)+'_'+str(year)+'.csv'
    # read in csv
    df = pd.read_csv(fileName)
    df['CVE_ENT']=df['CVE_ENT'].apply(fixEnt)
    df['CVE_MUN']=df['CVE_MUN'].apply(fixMun)
    df = deleteBadHists(df)
    if type(df)!=type(None):
        # get corr. yield data
        df = pd.merge(df, allYields[allYields['YEAR']==year],on=['CVE_ENT','CVE_MUN'])
        # Only keep rows w/ > 1000 ha sewn
        df['SEWN'] = df['SEWN'].apply(lambda x: float(x.replace(',','')))
        df = df[df['SEWN']>1000]
        if len(df.index) > 0:
            # return
            return(df)
        else:
            return(None)
    else:
        return(df)

allInputs = []
allOutputs = []
years = list(range(2003,2019))
states = list(range(1,33))
states = list(map(fixEnt,states))
for year in years:
    for state in states:
        combined = combineStateYear(state,year)
        # Add ouputs
        if type(combined)!=type(None):
            outputs = combined['YIELD'].tolist()
            for i in outputs:
                allOutputs.append(i)
            print(len(allOutputs))
            inputs = combined.apply(makeInput,axis=1)
            for i in inputs:
                allInputs.append(i)
            print(len(allInputs))
        else:
            print('None')

with open(processed_path+'input.pkl','wb') as outfile:
    pickle.dump(allInputs,outfile)
with open(processed_path+'outcome.pkl','wb') as outfile:
    pickle.dump(allOutputs,outfile)
