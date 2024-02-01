# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:44:22 2024

@author: rabni
"""

from datetime import datetime, timedelta
import ee
import geemap
import os
ee.Authenticate()
# Initialize the library.
ee.Initialize(project='sice-geus')

def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)



def add_one_day(original_date):
    
    # Convert the input string to a datetime object
    current_date = datetime.strptime(original_date, "%Y-%m-%d")
    # Add one day
    new_date = current_date + timedelta(days=1)
    # Format the result in "YYYY-MM-DD" format
    new_date_formatted = new_date.strftime("%Y-%m-%d")

    return new_date_formatted

def EarthEngine_S2(bounds,sday,id_tile):
    
    base_folder = os.path.abspath('..')

    # bounds_test = [[-49.5558500, 68.7588182],
    #                [-49.5435994, 68.7591537],
    #                [-49.5426584, 68.7547080],
    #                [-49.5548814, 68.7543593]]
    
    # Create a polygon geometry using the bounds
    polygon = ee.Geometry.Polygon(bounds)
    eday = add_one_day(sday)
    output_folder = base_folder + os.sep + 'output' + os.sep + 'S2_bio_tracker'
    data = (
      ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
      .filterBounds(polygon)
      .filterDate(sday, eday)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(mask_s2_clouds)
      # Pre-filter to get less cloudy granules.
    )
    
    if data.size().getInfo() == 0:
        return None
    
    # Select multiple bands
    selected_bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    selected_bands_col = ['B4','B3','B2']
    # Map over the image collection and select the desired bands
    selected_collection = data.map(lambda image: image.select(selected_bands))
    selected_collection_col = data.map(lambda image: image.select(selected_bands_col))
        
    selected_collection = selected_collection.first()
    selected_collection = ee.ImageCollection([selected_collection])
    
    selected_collection_col = selected_collection_col.first()
    selected_collection_col = ee.ImageCollection([selected_collection_col])
    
    data_repro = selected_collection.map(lambda image: image.reproject(crs="EPSG:3413", scale=20))
    data_repro_col = selected_collection_col.map(lambda image: image.reproject(crs="EPSG:3413", scale=20))
    
    geemap.ee_export_image_collection(data_repro, out_dir=output_folder, \
    region=polygon, scale=20, crs="EPSG:3413",file_per_band=True,filenames=[sday.replace('-','_') + f'_{id_tile}'])
        
    geemap.ee_export_image_collection(data_repro_col, out_dir=output_folder, \
    region=polygon, scale=20, crs="EPSG:3413",filenames=[sday.replace('-','_') + f'_{id_tile}_colour'])
    
    s2_ids = [output_folder + os.sep + sday.replace('-','_') + f'_{id_tile}.' + b + '.tif' for b in selected_bands]
            
    
    return s2_ids