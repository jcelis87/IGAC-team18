# region Imports
# OCR from Google Vision API
from google.cloud import vision

# Image processing
from rasterio.enums import Resampling
from PIL import Image, ImageFilter
from rasterio.mask import mask
from geotiff import GeoTiff
import rasterio
import cv2

# Data processing
import pandas as pd
import numpy as np

# GeoPandas
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
from shapely import geometry
import geopandas as gpd

# Manipulation of files in disk
import glob
import sys
import io
import os

# Other utilities
from imutils.object_detection import non_max_suppression
import time
import re
# endregion

def split_image_into_cells(img, filename, num_imgs=3):
    """
    Takes a Rasterio dataset and splits it into a grid of num_imgs x num_imgs.

    Arguments
    ---------
    img: rasterio.DatasetReader
         Image object to split
    
    filename: str
              Path of the image to split
    
    num_imgs: int
              Number of rows and columns to split the image into
    
    Output
    ------
    None
    """
    square_dim_wide = img.shape[1] // num_imgs
    square_dim_height = img.shape[0] // num_imgs
    
    number_of_cells_wide = img.shape[1] // square_dim_wide
    number_of_cells_high = img.shape[0] // square_dim_height
    x, y = 0, 0
    count = 0
    for hc in range(number_of_cells_high):
        y = hc * square_dim_height
        for wc in range(number_of_cells_wide):
            x = wc * square_dim_wide
            geom = get_tile_geom(img.transform, x, y, square_dim_wide, square_dim_height)
            get_cell_from_geom(img, geom, filename, count)
            count = count + 1

def get_tile_geom(transform, x, y, square_dim_wide, square_dim_height):
    """
    Generate a bounding box from the pixel-wise coordinates using the original 
    datasets transform property.

    Arguments
    ---------
    transform: rasterio.DatasetReader.transform object
               Transform object
    
    x: float
       X coordinate
    
    y: float
       Y coordinate
    
    squareDim_wide: int
                    Width dimension
    
    squareDim_height: int
                      Height dimension
    
    Output
    ------
    bounding_box: shapely.geometry.box
                  Bounding box object

    """
    corner1 = (x, y) * transform
    corner2 = (x + square_dim_wide, y + square_dim_height) * transform
    bounding_box = geometry.box(corner1[0], corner1[1],
                                corner2[0], corner2[1])    
    return bounding_box

def get_cell_from_geom(img, geom, filename, count):
    """
    Crop the dataset using the generated box and write it out as a GeoTIFF.

    Arguments
    ---------
    img: rasterio.DatasetReader
         Rasterio reprensentation of the image

    geom: shapely.geometry
          Geometry object

    filename: str
              Path of image file
    
    count: int
           Subimage index
    
    Output
    ------
    None
    """
    crop, cropTransform = mask(img, [geom], crop=True)
    write_image_as_GeoTIFF(crop,
                        cropTransform,
                        img.meta,
                        img.crs,
                        filename+"_"+str(count))

def write_image_as_GeoTIFF(img, transform, metadata, crs, filename):
    """
    Write the passed in dataset as a GeoTIFF.

    Arguments
    ---------
    img: rasterio.DatasetReader
         Rasterio representation of the image
    
    transform: rasterio.DatasetReader.transform
               Transform object
    
    metadata: rasterio.DatasetReader.meta
              Metadata of the file
    
    crs: rasterio.DatasetReader.crs
         Object representing the CRS system
    
    filename: str
              Path of the image file

    Output
    ------
    None
    """
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": 'EPSG:4686'})
    with rasterio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)

def detect_text(bytes_img):
    """
    Uses the Google Vision API to extract text from
    an image.
    
    Arguments
    ---------
    file_path: str
               path of the image to process.
    
    Outputs
    -------
    response: AnnotateImageResponse object
              json like format with bounding box and other
              relevant information.
    text: str
          text extracted from the image.
    """
    client = vision.ImageAnnotatorClient()    
    image = vision.Image(content=bytes_img)
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text
    
    return response, text

def getbytesimg_from_path(filepath):
    """
    Obtains the bytes base64 format of an image from a 
    local file path.
    
    Arguments
    ---------
    filepath: str
              Path of the image file to convert
    
    Output
    ------
    bytes_img: bytes
               base64 format of the image
    """
    with open(filepath, "rb") as image_file:
        bytes_img = image_file.read()
    
    return bytes_img

def split_images(filepath, num_imgs=3):
    """
    Split a large image into a grid of 3x3
    smaller images.
    
    Arguments:
    ---------
    filepath: str
              file path of the large image
    num_imgs: int (optional)
              Number of rows and columns of the grid
              
    Output
    ------
    None
    
    """
    img = rasterio.open(filepath)
    split_image_into_cells(img, "split_out/output_data")

def geotif_to_jpeg(tif_filename):
    """
    Converts geotif image from disk and saves it 
    in the same folder in jpeg format.
    
    Arguments
    ---------
    tif_filename: str
                  path of the tif file to convert
    
    Output
    ------
    None
    
    """
    with rasterio.open(tif_filename) as infile:    
        profile = infile.profile    
        profile['driver']='JPEG'
        jpeg_filename = tif_filename.replace(".tif", ".jpeg")
        with rasterio.open(jpeg_filename, 'w', **profile) as dst:
            dst.write(infile.read())

def get_text_coords(response):
    """
    Extracts the geographic names and bounding boxes coordinates
    from an AnnotateImageResponse object (Google).
    
    Arguments
    ---------
    response: AnnotateImageResponse (Google Vision API)
              Response object after calling the document_text_detection
              function
              
    Outputs
    -------
    palabras_google: list
                     List of strings containing the geographic names
    boundings_google: list
                      List of lists of 4 vertices that define the bounding 
                      box of each geographic name
    confidence_google: list
                       List of floats representing the confidence of the 
                       Google API at detecting each handwritten text
    """
    
    palabras_google = []
    boundings_google = []
    confidence_google = []
    
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            palabra_google = ''
            boundings_google.append(block.bounding_box)
            confidence_google.append(block.confidence)
            for parrafos in block.paragraphs:
                for palabras in parrafos.words:
                    for simbolo in palabras.symbols:
                        palabra_google = palabra_google+simbolo.text
                    palabra_google = palabra_google+' '
            palabras_google.append(palabra_google.rstrip())
            
    return palabras_google, boundings_google, confidence_google

def get_coords(boundings, word_index, point_index, axis):
    """
    Returns x or y coordinate from the list of vertices.
    
    Arguments
    ---------
    boundings: list
               List of bounding boxes with the vertices of 
               each bounding box
    word_index: int
                Index of the desired word starting at 0
    point_index: int
                 Index of the desired point (from 0 to 3)
    coord: str
           Either x or y
           
    Output
    ------
    coordinate: float
                Coordinate of the specified word, point, and axis
    
    """
    if axis == "x":
        coordinate = boundings[word_index].vertices[point_index].x
    else:
        coordinate = boundings[word_index].vertices[point_index].y
    
    return coordinate

def create_geometries(filepath):
    """
    Splits the image in a grid, uses Google Vision API to extract text
    and the bounding boxes. This method creates geojson files with 
    the geographic names, their respective geodesic coordinates, the 
    confidence, and the geometries of the bounding boxes.
    
    The geojson files are saved in the folder geometries.
    
    Arguments
    ---------
    filepath: str
              Path to the geotiff image file to process
              
    Output
    ------
    None
    """

    split_images(filepath)
    images_to_proces = os.listdir(path='split_out')

    for path in images_to_proces:
        if ('.tif' in path):
            geotif_to_jpeg("split_out/" + path)
            sub_image_jpeg = path.replace('.tif','.jpeg')
            response, text = detect_text(getbytesimg_from_path("split_out/" + sub_image_jpeg))
            words, boundings, confidence = get_text_coords(response)
            img = rasterio.open("split_out/" + sub_image_jpeg)

            # Generating the sub_geometries
            geometries = []
            centroids = []

            for i in range(len(words)):
                aux_polygon = Polygon([img.xy(get_coords(boundings,i,0,"y"),get_coords(boundings,i,0,"x")),
                                       img.xy(get_coords(boundings,i,1,"y"),get_coords(boundings,i,1,"x")),
                                       img.xy(get_coords(boundings,i,2,"y"),get_coords(boundings,i,2,"x")),
                                       img.xy(get_coords(boundings,i,3,"y"),get_coords(boundings,i,3,"x")),
                                       img.xy(get_coords(boundings,i,0,"y"),get_coords(boundings,i,0,"x"))])
                geometries.append(aux_polygon)
                centroids.append(aux_polygon.representative_point())

            sub_img_gdf = gpd.GeoDataFrame(columns=["toponimo_ocr","confidence",
                "centroide_longitud","centroide_latitud","geometry"], crs=str(img.crs))
            sub_img_gdf["geometry"] = geometries
            sub_img_gdf["toponimo_ocr"] = words
            sub_img_gdf["confidence"] = confidence
            sub_img_gdf["centroide_longitud"] = [x.coords[0][0] for x in centroids]
            sub_img_gdf["centroide_latitud"] = [x.coords[0][1] for x in centroids]
            name_geometry = 'geometries/'+sub_image_jpeg.replace('.jpeg','.geojson')
            try:
                sub_img_gdf.to_file(name_geometry, driver="GeoJSON")
            except:
                print('Empty text detected, no geometries generated!')

def combine_geometries(filepath):
    """
    Takes all the geojson files in the geometries folder and combine
    them into a single geojson file. It combines bounding boxes that 
    intersect each other and their respective geographic names. It 
    deletes the rows whose name has only numbers.
    
    Arguments
    ---------
    filepath: str
              Path of the original image
    
    Output
    ------
    all_toponyms_gdf: GeoDataFrame
                      GeoDataFrame with all the toponyms from the original
                      image. It contains the geographic name, geodesic 
                      coordinates of the centroid, confidence and the 
                      geometry of the bounding boxes.
    
    """
    org_img = rasterio.open(filepath)
    geometries_to_process = os.listdir(path='geometries')    
    rectangles = []
    all_toponyms_gdf = gpd.GeoDataFrame(columns=["toponimo_ocr","confidence",
                "centroide_longitud","centroide_latitud","geometry"],crs=str(org_img.crs))

    for geometry in geometries_to_process:
        if ".geojson" in geometry:
            file = gpd.read_file('geometries/' + geometry)
            all_toponyms_gdf = all_toponyms_gdf.append(file)

    all_toponyms_gdf.reset_index(drop=True, inplace=True)
    all_toponyms_gdf = geojson_posprocessing(all_toponyms_gdf)
    all_toponyms_gdf.to_file('text_detected.geojson',driver='GeoJSON')
    
    return all_toponyms_gdf

def get_image_corners(filepath):
    """
    Returns the geodesic coordinates of the original image.
    
    Arguments
    ---------
    filepath: str
              Path of the original image
    
    Output
    ------
    image_corners: dict
                   Dictionary with two keys: "upper_left" and "lower_left"
                   The values are tuples with the latitude and longitude
                   of the image corners.
    """
    original_img = rasterio.open(filepath)
    image_corners = {
        "upper_left": {
            "latitude": original_img.xy(0,0)[1],
            "longitude": original_img.xy(0,0)[0]
        },
        "lower_right": {
            "latitude": original_img.xy(original_img.shape[0], 
                                    original_img.shape[1])[1],
            "longitude": original_img.xy(original_img.shape[0], 
                                     original_img.shape[1])[0]
        }        
    }    
    return image_corners

def empty_folders():
    """
    Remove all temporary output files in the "split_out" and
    "geometries" folders. It also deletes the final output file
    detected_text.geojson.
    
    Arguments
    ---------
    None
    
    Output
    ------
    None
    """
    folders = ["split_out", "geometries"]
    
    for folder in folders:
        files = glob.glob(f"{folder}/*")
        for f in files:
            os.remove(f)
    try:
        os.remove("text_detected.geojson")
    except:
        print("text_detected.geojson already deleted!")

def geojson_posprocessing(geojson_toponyms):
    """
    Performs the following cleaning processes:
     - Identifies and combines polygons that intersect each other.
     - Combines the geographic names using the x coordinate 
    of the centroids to determine which name comes first. 
     - Remove punctuation symbols.
     - Delete the entries that do not have text
    
    Arguments
    ---------
    geojson_toponyms: gpd.GeoDataFrame
                      Initial GeoDataFrame union of all split images.
    
    Output: 
    ------
    new_geojson: GeoDataFrame
                 GeoDataFrame after processing the mentioned processing
    
    """
    
    new_geojson = gpd.GeoDataFrame(columns=["toponimo_ocr","confidence",
        "centroide_longitud","centroide_latitud","geometry"],
                                    crs=geojson_toponyms.crs)
    
    # Combine polygons that intersect each other
    while(len(geojson_toponyms) != 0):
        
        indexes = []
        new_elments =[]

        for i in range(0,len(geojson_toponyms)):
            if (geojson_toponyms.geometry[0].intersects(
                                geojson_toponyms.geometry[i])):
                indexes.append(i)

        if (len(indexes)==1):
            indexes = [0]
            new_geojson = new_geojson.append(geojson_toponyms.iloc[0:1])

        else:
            for j in range(0,4):
                for idx in range(len(indexes)):
                    for i in range(0,len(geojson_toponyms)):
                        if (geojson_toponyms.geometry[indexes[idx]].intersects(
                                                  geojson_toponyms.geometry[i])):
                            indexes.append(i)
                indexes = list(set(indexes))

            geom_list = [geojson_toponyms.geometry[j] for j in indexes]
            new_elments = [(geojson_toponyms['toponimo_ocr'][row],
                            geojson_toponyms['confidence'][row],
                            geojson_toponyms['centroide_longitud'][row]) for row in indexes]

            new_elments.sort(key=lambda tup: tup[2], reverse=False)
            toponimo_ocr = [' '.join([i[0] for i in new_elments])]
            confidence = [max([i[1] for i in new_elments])]
            new_geom = [cascaded_union(geom_list)]
            centroide_longitud = [new_geom[0].representative_point().coords[0][0]]
            centroide_latitud = [new_geom[0].representative_point().coords[0][1]]
            new_word = {'toponimo_ocr': toponimo_ocr,'confidence': confidence, 
                        'centroide_longitud': centroide_longitud,
                        'centroide_latitud': centroide_latitud,'geometry': new_geom}

            aux_geojson = gpd.GeoDataFrame(new_word,columns=["toponimo_ocr",
                            "confidence", "centroide_longitud","centroide_latitud", 
                                              "geometry"], crs=geojson_toponyms.crs)
            
            new_geojson = new_geojson.append(aux_geojson)

        geojson_toponyms = geojson_toponyms.drop(
            labels=indexes, axis=0).reset_index(drop=True)      
           
    new_geojson.reset_index(drop=True,inplace=True)
    
    # Remove punctuation
    new_geojson["toponimo_ocr"] = new_geojson["toponimo_ocr"].apply(
        lambda x: re.sub(r"[^\w\s]", "", x))
    
    # Remove rows that have a sequence of 3+ numbers
    no_numbers = new_geojson["toponimo_ocr"].apply(
        lambda x: False if re.search("\d{2,}", x) else True)
    new_geojson = new_geojson.loc[no_numbers].reset_index(drop=True)
        
    return new_geojson

def run_model1(filepath="img_to_process.tif"):
    """
    High level function that process the large image and returns
    a json file with all the toponyms, geometries of the bounding 
    boxes, confidence, and coordinates (latitude and longitude) of 
    each toponym.

    Arguments
    ---------
    filepath: str (optional)
              Path of the image to process. Default value is
              "img_to_process.tif". It needs a geotif image
              file that is properly georeferenced.

    Output
    ------
    results: str (geojson)
             GeoJSON represntation of the results as a string with the following structure:
             {
                 "type": "FeatureCollection",
                 "crs": { 
                    "type": "name", 
                    "properties": {
                         "name": "urn:ogc:def:crs:EPSG::4686" } },
                 "features": [{ 
                    "type": "Feature", 
                    "properties": { "toponimo_ocr": "SMEILIZAS", 
                                    "confidence": 0.75, 
                                    "centroide_longitud": -75.850919756153729,
                                    "centroide_latitud": 3.8895372826675358 }, 
                    "geometry": { "type": "Polygon", 
                                  "coordinates": [ [ 
                                      [ -75.852990469788153, 3.889918729916558 ], 
                                      [ -75.848821796287268, 3.889782498756193 ], 
                                      [ -75.848849042519305, 3.889155835418514 ], 
                                      [ -75.85301771602019, 3.889292066578879 ], 
                                      [ -75.852990469788153, 3.889918729916558 ] ] ] } },
              }
    """
    empty_folders()
    create_geometries(filepath)
    all_toponyms_gdf = combine_geometries(filepath)
    results = all_toponyms_gdf.to_json()

    return results