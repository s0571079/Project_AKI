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
midpoint = []
wma = []
mom = []
mfi = []
bop = []
adline = []
adosc = []
obv = []
atr = []
natr = []
tr = []
avgprice = []
typprice = []
wclprice = []
#htdcperiod = []
#htdcphase = []
whitesoldiers = []
starsinsouth = []
twocrows = []
linearreg = []
stddev = []
tsf = []

talib_values = numpy.empty([23, 200])

#Merge Chunks for Ta-Lib Calculations
ChunksMergedList = [allFilesAllChunks[0][0]]
i = 0
j = 0
while i < len(allFilesAllChunks):
    j = 0
    while j < len(allFilesAllChunks[i]):
        if i > 0 and j == 0:
            ChunksMergedList.append(allFilesAllChunks[i][j])
        if j > 0:
            ChunksMergedList[i] = ChunksMergedList[i].append(allFilesAllChunks[i][j], ignore_index=True)
        j = j + 1
    i = i + 1

# Calculate different parameters from different TA-LIB classes
for chunkDf in ChunksMergedList:
        # CLASS_1 : 'Overlap Studies' - Bollinger Bands (with Simple Moving Average)
        upper, middle, lower = TaLib_Calculations.getBBands(chunkDf)
        upper_df = pandas.DataFrame(data = upper)
        middle_df = pandas.DataFrame(data = middle)
        lower_df = pandas.DataFrame(data = lower)
        upperBB.append(upper_df)
        middleBB.append(middle_df)
        lowerBB.append(lower_df)
        #CLASS_1 : 'Overlap Studies' - Midpoint over Period
        midpoint.append(pandas.DataFrame(data = TaLib_Calculations.getMidpoint(chunkDf)))
        #CLASS_1 : 'Overlap Studies' - Weighted Moving Average
        wma.append(pandas.DataFrame(data =TaLib_Calculations.getWMA(chunkDf)))
        # CLASS_2 : 'Momentum Indicators' - Momentum
        mom.append(pandas.DataFrame(data =TaLib_Calculations.getMomentum(chunkDf)))
        # CLASS_2 : 'Momentum Indicators' - Money Flow Index
        mfi.append(pandas.DataFrame(data =TaLib_Calculations.getMFI(chunkDf)))
        # CLASS_2 : 'Momentum Indicators' - Balance of Power
        bop.append(pandas.DataFrame(data =TaLib_Calculations.getBOP(chunkDf)))
        # CLASS_3 : 'Volume Indicators' - Chaikin Accumulation/Distribution Line
        adline.append(pandas.DataFrame(data =TaLib_Calculations.getADLine(chunkDf)))
        # CLASS_3 : 'Volume Indicators' - Chaikin A/D Oscillator
        adosc.append(pandas.DataFrame(data =TaLib_Calculations.getADOscillator(chunkDf)))
        # CLASS_3 : 'Volume Indicators' - On Balance Volume
        obv.append(pandas.DataFrame(data =TaLib_Calculations.getOBV(chunkDf)))
        # CLASS_5 : 'Price Transform' - Average Price
        avgprice.append(pandas.DataFrame(data =TaLib_Calculations.getAvgPrice(chunkDf)))
        # CLASS_5 : 'Price Transform' - Typical Price
        typprice.append(pandas.DataFrame(data =TaLib_Calculations.getTypicalPrice(chunkDf)))
        # CLASS_5 : 'Price Transform' - Weighted Close Price
        wclprice.append(pandas.DataFrame(data =TaLib_Calculations.getWClPrice(chunkDf)))
        # CLASS_6 : 'Volatility Indicators - Average True Range
        atr.append(pandas.DataFrame(data =TaLib_Calculations.getAverageTrueRange(chunkDf)))
        # CLASS_6 : 'Volatility Indicators - Normalized Average True Range
        natr.append(pandas.DataFrame(data =TaLib_Calculations.getNATR(chunkDf)))
        # CLASS_6 : 'Volatility Indicators - True Range (NaN an erster Stelle rausfilern?)
        tr.append(pandas.DataFrame(data = TaLib_Calculations.getTrueRange(chunkDf)))
        # CLASS_7 : 'Pattern Recognition' - Three Advanced White Soldiers
        whitesoldiers.append(pandas.DataFrame(data =TaLib_Calculations.get3AWS(chunkDf)))
        # CLASS_7 : 'Pattern Recognition' - Three Stars in the South
        starsinsouth.append(pandas.DataFrame(data =TaLib_Calculations.get3SITS(chunkDf)))
        # CLASS_7 : 'Pattern Recognition' - Two Crows
        twocrows.append(pandas.DataFrame(data =TaLib_Calculations.get2Crows(chunkDf)))
        # CLASS_8 : 'Statistic Functions' - Linear Regression
        linearreg.append(pandas.DataFrame(data =TaLib_Calculations.getLinearReg(chunkDf)))
        # CLASS_8 : 'Statistic Functions' - Standard Deviation
        stddev.append(pandas.DataFrame(data =TaLib_Calculations.getStdDev(chunkDf)))
        # CLASS_8 : 'Statistic Functions' - Time Series Forecast
        tsf.append(pandas.DataFrame(data =TaLib_Calculations.getTSF(chunkDf)))


print("TALIB DATA CALCULATION finished ...")

# Normalize Talib Data
columns = numpy.array(['upperBB','middleBB','lowerBB','midpoint','wma','mom','mfi','bop','adline',
           'adosc','obv','atr','natr','tr','avgprice','typprice','wclprice',
           'whitesoldiers','starsinsouth','twocrows','linearreg','stddev','tsf'])

TaLib_list = [upperBB, middleBB, lowerBB, midpoint, wma, mom, mfi, bop, adline, adosc, obv, atr, natr, tr, avgprice,
                          typprice, wclprice, whitesoldiers, starsinsouth, twocrows, linearreg, stddev, tsf]

j = 0
for method in TaLib_list:
    i = 0
    for df in method:
        min_max_scaler = preprocessing.MinMaxScaler()
        x = df.values
        x_scaled = min_max_scaler.fit_transform(x)
        table = pandas.DataFrame(x_scaled)
        if j == 0:
            upperBB[i] = table
            i = i + 1
        elif j == 1:
            middleBB[i] = table
            i = i + 1
        elif j == 2:
            lowerBB[i] = table
            i = i + 1
        elif j == 3:
            midpoint[i] = table
            i = i + 1
        elif j == 4:
            wma[i] = table
            i = i + 1
        elif j == 5:
            mom[i] = table
            i = i + 1
        elif j == 6:
            mfi[i] = table
            i = i + 1
        elif j == 7:
            bop[i] = table
            i = i + 1
        elif j == 8:
            adline[i] = table
            i = i + 1
        elif j == 9:
            adosc[i] = table
            i = i + 1
        elif j == 10:
            obv[i] = table
            i = i + 1
        elif j == 11:
            atr[i] = table
            i = i + 1
        elif j == 12:
            natr[i] = table
            i = i + 1
        elif j == 13:
            tr[i] = table
            i = i + 1
        elif j == 14:
            avgprice[i] = table
            i = i + 1
        elif j == 15:
            typprice[i] = table
            i = i + 1
        elif j == 16:
            wclprice[i] = table
            i = i + 1
        elif j == 17:
            whitesoldiers[i] = table
            i = i + 1
        elif j == 18:
            starsinsouth[i] = table
            i = i + 1
        elif j == 19:
            twocrows[i] = table
            i = i + 1
        elif j == 20:
            linearreg[i] = table
            i = i + 1
        elif j == 21:
            stddev[i] = table
            i = i + 1
        elif j == 22:
            tsf[i] = table
            i = i + 1
        else:
            print('Something is wrong...')
    j = j + 1


#Merge normalized Chunks and Combine with Ta-Lib Data
ChunksMergedList_validated = [allFilesAllChunks_validated1[0][0]]
i = 0
j = 0
while i < len(allFilesAllChunks_validated1):
    j = 0
    while j < len(allFilesAllChunks_validated1[i]):
        if i > 0 and j == 0:
            ChunksMergedList_validated.append(allFilesAllChunks_validated1[i][j])
        if j > 0:
            ChunksMergedList_validated[i] = ChunksMergedList_validated[i].append(allFilesAllChunks_validated1[i][j], ignore_index=True)
        j = j + 1
    i = i + 1

columns = numpy.array(['Open','High','Low','Close','Volume','Date','Ticker',
                       'upperBB','middleBB','lowerBB','midpoint','wma','mom','mfi','bop','adline',
                       'adosc','obv','atr','natr','tr','avgprice','typprice','wclprice',
                       'whitesoldiers','starsinsouth','twocrows','linearreg','stddev','tsf'])

i = 0
allFilesAllData = []
while i < 10:
    allFilesAllData.append(pandas.concat([ChunksMergedList_validated[i], upperBB[i], middleBB[i], lowerBB[i], midpoint[i],
                                          wma[i], mom[i], mfi[i], bop[i], adline[i], adosc[i], obv[i], atr[i], natr[i], tr[i], avgprice[i],
                                          typprice[i], wclprice[i], whitesoldiers[i], starsinsouth[i], twocrows[i], linearreg[i], stddev[i], tsf[i]],
                                         axis=1))
    i = i + 1
for dataFrames in allFilesAllData:
    dataFrames.columns = columns

allFilesAllDataNoNans = []
for dataFrames in allFilesAllData:
    temp_df = dataFrames.dropna()
    temp_df.index = range(len(temp_df))
    allFilesAllDataNoNans.append(temp_df)
print("DATA NORMALIZATION AND MERGING finished...")

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