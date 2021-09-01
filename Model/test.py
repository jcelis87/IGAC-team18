import model1

#results = model1.run_model1("geotiffs/C-1974 F-238.tif")
img_corners = model1.get_image_corners("geotiffs/C-1974 F-238.tif")

print(img_corners)