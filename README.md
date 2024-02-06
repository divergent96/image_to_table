# Image to table converter

Little side project iof mine to simplify budgeting hurdles. Reads images to parse relevanmt information into a table.  


## Problem Statement
1. Identify appropriate libraries for reading pdfs and parsing images
1. Learn and apply ocr methods to convert images to some readable table

## Data
Not uploaded for privacy.

## Initialization
1. Use conda create -p venv python=3.xx -y to initialize a new env
1. Use pip install -r requirements.txt to make required environment. (REMINDER IN CASE IT GETS MISSED) 

## Approach
1. Setup logging and exception handling structure
1. Import pdf file and read 1 page
1. Implement OCR to get some form of text
1. Parse OCR based text and apply logic to convert to table/seperate inforamtion out
1. Create table
1. Repeat for other pages
1. Merge all page outputs
1. Handle exceptions
1. FUTURE ADDONS

## Reference - General ML lifecycle

1. Define problem statement
1. Collect Data
1. High-level data check for discrepancies (missing entries, scrap data columns)
1. Data Exploration (EDA)
1. Pre-processing data
1. Train models
1. Model selection
