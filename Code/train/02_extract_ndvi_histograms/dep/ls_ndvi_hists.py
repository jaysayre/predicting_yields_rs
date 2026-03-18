from ast import literal_eval
import os
import re
import sys

import pandas as pd

import ee

ndvi_min = float(sys.argv[1]) # NDVI below this value goes into first bin
ndvi_max = float(sys.argv[2]) # NDVI above this value goes into last bin
gcvi_min = float(sys.argv[3]) # GCVI below this value goes into first bin
gcvi_max = float(sys.argv[4]) # GCVI above this value goes into last bin
ndti_min = float(sys.argv[5]) # NDTI below this value goes into first bin
ndti_max = float(sys.argv[6]) # NDTI above this value goes into last bin
bins = int(sys.argv[7]) # Number of bins in 1-D histogram
agg_func = sys.argv[8] # function to aggregate NDVI across images ('max','med','mean')

ee.Initialize()


def scale_facts(img):
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return(img.addBands(opticalBands,overwrite=True).addBands(thermalBands,overwrite=True))


def assign_bin(img,min,max,bins):
    # Bottom censor image
    img = img.max(min+((max-min)/1_000_000))
    # Top-censor image
    img = img.min(max-((max-min)/1_000_000))
    # Subtract min
    img = img.subtract(min)
    # Divide by binsize and floor
    bin = img.divide((max-min)/bins).floor()

    return(bin)

def calc_1d_bin(binned, bins):
    '''
    binned: an image with four bands of bins
    bins: the number of bins
    '''
    out = binned.select(0)
    out = out.add(binned.select(1).multiply(bins))
    out = out.rename("hist")
    return(out)

def cloud_mask(img):
    '''
    Function to mask cloudy pixels in a ls image using QA_PIXEL bits. Note: the relevant bits of QA_PIXEL are the same across satellites.

    Inputs:
        - img (ee.Image): the ls image to mask

    Outputs:
        - mask_img (ee.Image): img with cloudy pixels masked
    '''

    qa = img.select(['QA_PIXEL'])

    mask = ee.Image.constant(1).subtract(qa.bitwiseAnd(1<<3).And(qa.bitwiseAnd(1<<9)).Or(qa.bitwiseAnd(1<<4)))

    return(img.updateMask(mask))

def calc_ndvi(img,red,nir):
    '''
    Function to calculate NDVI in an image

    Inputs:
        - img (ee.Image): The ls image to calculate on
        - red (str): red band name
        - nir (str): nir band name
    Outputs:
        - ndvi (ee.Image): image with one ndvi band named 'ndvi'
    '''
    img = scale_facts(img)
    img = cloud_mask(img)
    # NDVI = (NIR-Red) / (NIR+Red)
    ndvi = img.normalizedDifference([nir,red]).rename('ndvi')

    return(ndvi)

def calc_gcvi(img,nir,green):
    img = scale_facts(img)
    img = cloud_mask(img)
    gcvi = ((img.select(nir).divide(img.select(green))).subtract(1)).rename('gcvi')
    return(gcvi)


def calc_ndti(img,swir1,swir2):
    img = scale_facts(img)
    img = cloud_mask(img)
    ndti = img.normalizedDifference([swir1,swir2]).rename('ndti')
    return(ndti)

def add_zeros(x,n):
    x = str(x)
    while len(x)<n:
        x = "0"+x
    return(x)



# Get Municipalities into GEE
munis = pd.read_csv("../processed_data/munis_agmask.csv")
munis['coords'] = munis['coords'].apply(literal_eval)
munis['eeGeom'] = munis['coords'].apply(ee.Geometry)
#munis['CVE_ENT'] = munis['CVE_ENT'].apply(lambda x: add_zeros(x,2))
#munis['CVE_MUN'] = munis['CVE_MUN'].apply(lambda x: add_zeros(x,3))
#munis['muncode'] = (munis['CVE_ENT']+munis['CVE_MUN']).apply(int)
#maize_munis = pd.read_csv("../processed_data/maize_munis.csv")

#munis = pd.merge(munis,maize_munis[['muncode']],
#                 on = ['muncode'])

peak_dates = pd.read_csv("../processed_data/muni_ndvi_peak_dates.csv")
peak_dates['CVE_ENT'] = peak_dates['CVE_ENT'].apply(lambda x: add_zeros(x,2))
peak_dates['CVE_MUN'] = peak_dates['CVE_MUN'].apply(lambda x: add_zeros(x,3))
peak_dates['muncode'] = (peak_dates['CVE_ENT']+peak_dates['CVE_MUN']).apply(int)

munis = pd.merge(munis,peak_dates,
                 on = ['muncode'])



munis['eeFeature'] = munis.apply(lambda x: ee.Feature(x['eeGeom'],
                                                      {'muncode':x['muncode'],
                                                       'start_p1':x['start_p1'],
                                                       'end_p1':x['end_p1'],
                                                       'start_p2':x['start_p2'],
                                                       'end_p2':x['end_p2']}),axis=1)



print("Municipalities Uploaded")

n = 25






def muni_ndvi(muni,start_date,end_date,agg_func):
    
    
    ls5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(muni.geometry()).filterDate(start_date,end_date)

    ls7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(muni.geometry()).filterDate(start_date,end_date)

    ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(muni.geometry()).filterDate(start_date,end_date)

    ls5_ndvi = ee.Algorithms.If(ls5.size(),
                                ls5.map(lambda x: calc_ndvi(x,'SR_B3','SR_B4')),
                                ee.ImageCollection([]))
    ls5_ndvi = ee.ImageCollection(ls5_ndvi)
    ls7_ndvi = ee.Algorithms.If(ls7.size(),
                                ls7.map(lambda x: calc_ndvi(x,'SR_B3','SR_B4')),
                                ee.ImageCollection([]))
    ls7_ndvi = ee.ImageCollection(ls7_ndvi)
    ls8_ndvi = ee.Algorithms.If(ls8.size(),
                                ls8.map(lambda x: calc_ndvi(x,'SR_B4',"SR_B5")),
                                ee.ImageCollection([]))
    ls8_ndvi = ee.ImageCollection(ls8_ndvi)
    
    all_ndvi = ls5_ndvi.merge(ls7_ndvi).merge(ls8_ndvi)
    all_ndvi = ee.ImageCollection(all_ndvi)

    all_ndvi = ee.Algorithms.If(all_ndvi.size(),
                                all_ndvi,
                                ee.ImageCollection([ee.Image.constant(0)])
    )
    all_ndvi = ee.ImageCollection(all_ndvi)

    if agg_func=="mean":
        all_ndvi = all_ndvi.mean()

    elif agg_func=="med":
        all_ndvi = all_ndvi.median()

    elif agg_func=="min":
        all_ndvi = all_ndvi.min()

    elif agg_func=="max":
        all_ndvi = all_ndvi.max()
    else:
        raise Exception("agg_func needs to be 'max', 'med', or 'mean'")


    ls5_gcvi = ee.Algorithms.If(ls5.size(),
                                ls5.map(lambda x: calc_gcvi(x,'SR_B4','SR_B2')),
                                ee.ImageCollection([]))
    ls5_gcvi = ee.ImageCollection(ls5_gcvi)
    ls7_gcvi = ee.Algorithms.If(ls7.size(),
                                ls7.map(lambda x: calc_gcvi(x,'SR_B4','SR_B2')),
                                ee.ImageCollection([]))
    ls7_gcvi = ee.ImageCollection(ls7_gcvi)
    ls8_gcvi = ee.Algorithms.If(ls8.size(),
                                ls8.map(lambda x: calc_gcvi(x,'SR_B5',"SR_B3")),
                                ee.ImageCollection([]))
    ls8_gcvi = ee.ImageCollection(ls8_gcvi)
    
    all_gcvi = ls5_gcvi.merge(ls7_gcvi).merge(ls8_gcvi)
    all_gcvi = ee.ImageCollection(all_gcvi)

    all_gcvi = ee.Algorithms.If(all_gcvi.size(),
                                all_gcvi,
                                ee.ImageCollection([ee.Image.constant(0)])
    )
    all_gcvi = ee.ImageCollection(all_gcvi)
    all_gcvi = all_gcvi.max()


    ls5_ndti = ee.Algorithms.If(ls5.size(),
                                ls5.map(lambda x: calc_ndti(x,'SR_B5','SR_B7')),
                                ee.ImageCollection([]))
    ls5_ndti = ee.ImageCollection(ls5_ndti)
    ls7_ndti = ee.Algorithms.If(ls7.size(),
                                ls7.map(lambda x: calc_ndti(x,'SR_B5','SR_B7')),
                                ee.ImageCollection([]))
    ls7_ndti = ee.ImageCollection(ls7_ndti)
    ls8_ndti = ee.Algorithms.If(ls8.size(),
                                ls8.map(lambda x: calc_ndti(x,'SR_B6',"SR_B7")),
                                ee.ImageCollection([]))
    ls8_ndti = ee.ImageCollection(ls8_ndti)
    
    all_ndti = ls5_ndti.merge(ls7_ndti).merge(ls8_ndti)
    all_ndti = ee.ImageCollection(all_ndti)

    all_ndti = ee.Algorithms.If(all_ndti.size(),
                                all_ndti,
                                ee.ImageCollection([ee.Image.constant(-100)])
    )
    all_ndti = ee.ImageCollection(all_ndti)
    all_ndti = all_ndti.min()

    all_vars = ee.Image(all_ndvi).addBands(ee.Image(all_gcvi))
    all_vars = all_vars.addBands(ee.Image(all_ndti))
    return(all_vars)

def muni_ndvi_year(muni):
    p1_ndvi = muni_ndvi(muni,muni.get('start_p1'),muni.get('end_p1'),agg_func)
    p1_ndvi = ee.Image(p1_ndvi).rename(['ndvi_p1','gcvi_p1','ndti_p1'])
    p2_ndvi = muni_ndvi(muni,muni.get('start_p2'),muni.get('end_p2'),agg_func)
    p2_ndvi = ee.Image(p2_ndvi).rename(['ndvi_p2','gcvi_p2','ndti_p2'])
    both_ndvi = p1_ndvi.addBands(p2_ndvi)
    return(both_ndvi)

def muni_hist(muni,bins):

    img = muni_ndvi_year(muni)
    muni_p1 = assign_bin(img.select('ndvi_p1'),ndvi_min,ndvi_max,bins)
    muni_p2 = assign_bin(img.select('ndvi_p2'),ndvi_min,ndvi_max,bins)

    binned_ndvi = muni_p1.addBands(muni_p2)

    binned_1d = calc_1d_bin(binned_ndvi, bins).rename('ndvi_hist')

    gcvi_p1 = assign_bin(img.select('gcvi_p1'),gcvi_min,gcvi_max,bins)
    gcvi_p2 = assign_bin(img.select('gcvi_p2'),gcvi_min,gcvi_max,bins)

    binned_gcvi = gcvi_p1.addBands(gcvi_p2)
    gcvi_1d = calc_1d_bin(binned_gcvi, bins).rename('gcvi_hist')

    ndti_p1 = assign_bin(img.select('ndti_p1'),ndti_min,ndti_max,bins)
    ndti_p2 = assign_bin(img.select('ndti_p2'),ndti_min,ndti_max,bins)

    binned_ndti = ndti_p1.addBands(ndti_p2)
    ndti_1d = calc_1d_bin(binned_ndti, bins).rename('ndti_hist')

    binned_1d = binned_1d.addBands(gcvi_1d).addBands(ndti_1d)

    hist = binned_1d.reduceRegion(
        ee.Reducer.fixedHistogram(0,bins**2,bins**2),
        muni.geometry(),
        30,
        maxPixels=500_000_000
    )

    return(muni.set(hist))

cols = ["muncode","ndvi_hist","gcvi_hist","ndti_hist"]

ents = munis['CVE_ENT'].unique()
for e in range(0,len(ents)):
    ent = ents[e]

    for year in range(2003,2025):

        munis_year = munis.loc[munis['year']==year]
        
        for i in range(0,(len(munis_year.loc[munis_year['CVE_ENT']==ent].index)//n)+1):

            ee_munis = ee.FeatureCollection(munis_year.loc[munis_year['CVE_ENT']==ent,'eeFeature'].iloc[i*n:min((i+1)*n,len(munis_year.loc[munis_year['CVE_ENT']==ent].index)+1)].to_list())

        
            print(f"State: {ent}, Group: {i}, Year: {year}")
            munis_ndvi_yr = ee_munis.map(lambda x: muni_hist(x,bins))

            task = ee.batch.Export.table.toDrive(collection=munis_ndvi_yr,
                                         folder=f'muni_vi_hists_{ndvi_min}_{ndvi_max}_{gcvi_min}_{gcvi_max}_{ndti_min}_{ndti_max}_{bins}_{agg_func}',description = f'muni_ndvi_hist_{ent}_{i}_{year}',
                                         selectors=cols)
            
            task.start()