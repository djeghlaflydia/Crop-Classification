# test_data.py
import ee

# Initialize with your project
ee.Initialize(project='crop-classification-2021')

print("Testing data access...\n")

# Test 1: CDL Data
try:
    cdl = ee.Image('USDA/NASS/CDL/2021')
    print("✅ CDL 2021 is accessible")
except Exception as e:
    print(f"❌ CDL error: {e}")

# Test 2: Sentinel-2 Data
try:
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(5)
    count = s2.size().getInfo()
    print(f"✅ Sentinel-2 accessible (found {count} images)")
except Exception as e:
    print(f"❌ S2 error: {e}")

# Test 3: Get info about your study area (California)
try:
    california_bbox = [-120.175, 36.725, -120.125, 36.775]
    geometry = ee.Geometry.BBox(*california_bbox)
    
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(geometry)
                     .filterDate('2021-06-01', '2021-09-30')
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    count = s2_collection.size().getInfo()
    print(f"✅ California study area: {count} cloud-free S2 images")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎉 Earth Engine is ready for your crop classification project!")