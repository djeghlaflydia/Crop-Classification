"""
api_access.py - Google Earth Engine data access module
"""

import ee
import numpy as np
import os
from typing import List, Tuple, Optional

def init_gee(project: str = None):
    """Initialize Google Earth Engine API"""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize GEE: {e}")
        print("Please run: earthengine authenticate")
        return False

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds and shadows in Sentinel-2 imagery using QA band.
    CORRECTED VERSION - no Reducer.any() issue
    """
    # QA60 band: bit10 = cloud, bit11 = cirrus
    qa = image.select('QA60')
    
    # Cloud mask bits
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Clear pixels: both cloud and cirrus bits are 0
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Apply mask and scale reflectance to [0,1]
    return image.updateMask(mask).divide(10000)

def get_s2_collection(bbox: List[float], start_date: str, end_date: str, 
                      max_cloud: int = 20) -> ee.ImageCollection:
    """Get Sentinel-2 image collection for a bounding box"""
    geometry = ee.Geometry.BBox(*bbox)
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
                  .map(mask_s2_clouds))
    
    return collection

def get_cdl_image(bbox: List[float], year: int) -> ee.Image:
    """Get Cropland Data Layer image for a given year"""
    geometry = ee.Geometry.BBox(*bbox)
    
    cdl = (ee.Image(f'USDA/NASS/CDL/{year}')
           .clip(geometry)
           .select('cropland'))
    
    return cdl

def compute_ndvi(image: ee.Image) -> ee.Image:
    """Compute NDVI (Normalized Difference Vegetation Index)"""
    nir = image.select('B8')
    red = image.select('B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return image.addBands(ndvi)

def compute_evi(image: ee.Image) -> ee.Image:
    """Compute EVI (Enhanced Vegetation Index)"""
    nir = image.select('B8')
    red = image.select('B4')
    blue = image.select('B2')
    
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename('EVI')
    
    return image.addBands(evi)

def compute_ndwi(image: ee.Image) -> ee.Image:
    """Compute NDWI (Normalized Difference Water Index)"""
    green = image.select('B3')
    nir = image.select('B8')
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    return image.addBands(ndwi)

def get_s2_collection_with_indices(bbox: List[float], start_date: str, 
                                    end_date: str, max_cloud: int = 20) -> ee.ImageCollection:
    """Get Sentinel-2 collection with vegetation indices"""
    collection = get_s2_collection(bbox, start_date, end_date, max_cloud)
    collection = collection.map(compute_ndvi)
    collection = collection.map(compute_evi)
    collection = collection.map(compute_ndwi)
    return collection

def create_10day_composite(collection: ee.ImageCollection, 
                           start_date: str, 
                           end_date: str) -> ee.Image:
    """Create a 10-day median composite from an image collection"""
    filtered = collection.filterDate(start_date, end_date)
    composite = filtered.median()
    return composite