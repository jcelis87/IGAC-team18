# IGAC-team18
This repository contains the code related to:
- The model for text recognition
- Geo database

The model is divided into three main stages:
1. Split a large image into smaller images keeping the coordinates
2. Use the `east` algorithm on each small image to detect the regions where there is text keeping the coordinates
3. Split the small image into a set of smaller images containing only text and calculating a point coordinate for the toponym
4. Run an OCR algorithm (from pytesseract) to convert the smallest images into text
5. Return the toponyms and coordinates
