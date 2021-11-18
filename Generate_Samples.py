"""
This file is used to generate the pickle files in the desired format to feed it to the neuronal network.

Steps which happen here:

READING DATA
- Find all available CSV stock files
- Open each csv. file
- Read each file from the top to the bottom step by step and build chunks based on a given chunk size (we take 22 here)
- Validate each result per file (files with less than 22 rows are ignored)

VALIDATION AND PREPROCESSING
- Delete not needed columns
- Filtering some conditions (filter out null values, filter out chunks with no variation)
- Validation
- Normalisation

ENRICH DATA WITH TA-LIB PARAMETERS
- (started) Calculate different parameters from different TA-LIB classes (https://github.com/mrjbq7/ta-lib)
- (missing) Add the results to each chunk row

PERSIST VALUES
- (missing) Save the preprocessed data as pickle files


???
- Volume deleted

"""

import os
import pandas
import numpy
import talib
import csv
import pickle
import TaLib_Calculations
from sklearn import preprocessing

# basic settings
data_folder_path = "c:/data/USWS_Subset"
pickle_files_folder_path = "missing"

# loop settings
numberOfFilesToRead = 10
chunksPerFileToRead = 20
chunkSize = 22 # size of rows in one chunk

# init variables
allFilesAllChunks = []
numberOfFilesReadingFinished = 0

for root, dirs, files in os.walk(data_folder_path):

    print("Found stock files to read:" + str(len(files)))

    for file in files:
        if file.endswith(".csv"):
            allChunksInFile = []
            allRowsSingleChunk = []

            f = open(data_folder_path + "/" + file, 'r')
            reader = pandas.read_csv(f)

            startIndex = 0
            should_restart = True
            while should_restart:
                should_restart = False
                for index, row in reader.iterrows():
                    if index >= startIndex:
                        allRowsSingleChunk.append(row)
                        if len(allRowsSingleChunk) == chunkSize:
                            dataFrame = pandas.DataFrame(allRowsSingleChunk)
                            #dataFrame.drop('Volume', axis=1, inplace=True)
                            allChunksInFile.append(pandas.DataFrame(dataFrame))
                            allRowsSingleChunk.clear()
                            startIndex = startIndex + 1
                            should_restart = True
                            if len(allChunksInFile) == chunksPerFileToRead:
                                should_restart = False
                            break

            # here we make sure, the data is only added if the file had enough rows (>= chunkSize)
            if len(allChunksInFile) > 1:
                allFilesAllChunks.append(list(allChunksInFile))
                numberOfFilesReadingFinished = numberOfFilesReadingFinished + 1
                print("Files read:" + str(numberOfFilesReadingFinished))
            else:
                print("File hat nicht genügend Einträge (Chunksize)" + file)
            if numberOfFilesToRead == numberOfFilesReadingFinished:
                break

        f.close()

print("READING DATA finished ..." + str(numberOfFilesReadingFinished) + " files are read successfully")

# Filter out null values and ignore these chunks
# Filter out values with no variation
# Perform normalization
allFilesAllChunks_validated1 = []
for allChunksSingleFile in allFilesAllChunks:
    allChunksSingleFileValidated = []
    for chunkDf in allChunksSingleFile:
        # Filtering several conditions happens here
        chunkDf.dropna(inplace=True)
        numberOfDifferentValuesSeries = chunkDf.nunique()
        countDiffValuesOpen = numberOfDifferentValuesSeries['Open']
        countDiffValuesClose = numberOfDifferentValuesSeries['Close']
        if (len(chunkDf) == chunkSize) and (countDiffValuesOpen != 1) and (countDiffValuesClose != 1):
            # validation passed
            # perform normalization of columns
            min_max_scale = preprocessing.MinMaxScaler()
            # we need to remove the non semantic columns first (date & name)
            chunkDfAsArray = chunkDf.values[:, 1:-1]
            normalizedDf = min_max_scale.fit_transform(chunkDfAsArray)
            # add the non semantic columns again
            chunkDfPrepared = pandas.DataFrame(normalizedDf, columns=chunkDf.columns[1:-1])
            chunkDfPrepared['Date'] = chunkDf['Date'].values
            chunkDfPrepared['Ticker'] = chunkDf['Ticker'].values
            allChunksSingleFileValidated.append(pandas.DataFrame(chunkDfPrepared))
        else:
            print("NULL-Validation or UNIQUE-Validation not passed! Values: " + str(len(chunkDf)) + " / " + str(countDiffValuesOpen) + " / " + str(countDiffValuesClose))
    allFilesAllChunks_validated1.append(list(allChunksSingleFileValidated))

print("VALIDATION AND PREPROCESSING finished ...")

#init lists
upperBB = []
middleBB = []
lowerBB = []
dema = []
mom = []
adline = []
atr = []
avgprice = []
#htdcperiod = []
#htdcphase = []
whitesoldiers = []
starsinsouth = []
twocrows = []
linearreg = []

# Calculate different parameters from different TA-LIB classes
for allChunksSingleFile in allFilesAllChunks_validated1:
    for chunkDf in allChunksSingleFile:
        # CLASS_1 : 'Overlap Studies' - Bollinger Bands (with Simple Moving Average)
        upper, middle, lower = TaLib_Calculations.getBBands(chunkDf)
        upperBB.append(upper[21])
        middleBB.append(middle[21])
        lowerBB.append(lower[21])
        # CLASS_2 : 'Overlap Studies' - Double Exponential Moving Average
        demas = TaLib_Calculations.getDEMA(chunkDf)
        dema.append(demas[~numpy.isnan(demas)])
        # CLASS_3 : 'Momentum Indicators' - Momentum
        mom.append(TaLib_Calculations.getMomentum(chunkDf)[21])
        # CLASS_4 : 'Volume Indicators' - Chaikin Accumulation/Distribution Line
        adline.append(TaLib_Calculations.getADLine(chunkDf))
        # CLASS_5 : 'Cycle Indicators' - Hilbert Transform - Dominant Cycle Period and Dominant Cycle Phase (DO NOT WORK)
        #htdcperiod.append(TaLib_Calculations.getHTDCPeriod(chunkDf))
        #htdcphase.append(TaLib_Calculations.getHTDCPhase(chunkDf))
        # CLASS_6 : 'Price Transform' - Average Price
        avgprice.append(TaLib_Calculations.getAvgPrice(chunkDf))
        # CLASS_7 : 'Volatility Indicators
        atr.append(TaLib_Calculations.getAverageTrueRange(chunkDf)[21])
        # CLASS_8 : 'Pattern Recognition'
        whitesoldiers.append(TaLib_Calculations.get3AWS(chunkDf))
        starsinsouth.append(TaLib_Calculations.get3SITS(chunkDf))
        twocrows.append(TaLib_Calculations.get2Crows(chunkDf))
        # CLASS_9 : 'Statistic Functions' - Linear Regression
        linearreg.append(TaLib_Calculations.getLinearReg(chunkDf)[21])

# Save as pickle file
"""
    # Wir gehen jetzt durch die Spalten
    for col in correlations.columns:
        item = dict()
        item['y'] = targets[col]
        # Alle Y Werte als input
        item['Y'] = table[col]
        pos_stocks = list(correlations[col].nlargest(21).index)  # largest correlation is with stock itself
        pos_stocks.remove(col)
        item['X_p'] = table[pos_stocks]
        neg_stocks = list(correlations[col].nsmallest(20).index)
        item['X_n'] = table[neg_stocks]
        with open('c:/data/htw/2021_SS/AKI/Samples/' + col + '_' + str(time) + '.pkl', 'wb') as f:
            pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
    time = time + 1
"""