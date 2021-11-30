"""
This file is used to generate the pickle files in the desired format to feed it to the neuronal network.
For visualisation see './Grafiken/Basic_procedure.png'

Steps which happen here:

READING DATA
- Find all available CSV stock files
- Open each csv. file
- Read each file from the top to the bottom step by step and build chunks based on a given chunk size (we take 22 here)
- Validate each result per file (files with less than 22 rows are ignored)

VALIDATION AND PREPROCESSING
- Delete not needed columns
- Normalisation

ENRICH DATA WITH TA-LIB PARAMETERS
- Calculate different parameters from different TA-LIB classes (https://github.com/mrjbq7/ta-lib)
- Add the results to each row

PERSIST VALUES
- Save the preprocessed data as pickle files
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

numberOfFilesToRead = 600

chunksPerFileToRead = 20
chunkSize = 23 # size of rows in one chunk

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
# Perform normalization
normalized_data = []
allFilesMergedChunks = []
for allChunksSingleFile in allFilesAllChunks:
    allFilesMergedChunks.append(pandas.concat(allChunksSingleFile))
    for dataFrame in allFilesMergedChunks:
        min_max_scaler = preprocessing.MinMaxScaler()
        x = dataFrame.dropna().values[:, 1:-1]
        x_scaled = min_max_scaler.fit_transform(x)
        table = pandas.DataFrame(x_scaled)
    normalized_data.append(table)

allFilesMergedChunks_normalized = []
for df in allFilesMergedChunks:
    allFilesMergedChunks_normalized.append(df.copy())

for i in range(len(allFilesMergedChunks)):
    allFilesMergedChunks_normalized[i]['Open'] = normalized_data[i][0]
    allFilesMergedChunks_normalized[i]['High'] = normalized_data[i][1]
    allFilesMergedChunks_normalized[i]['Low'] = normalized_data[i][2]
    allFilesMergedChunks_normalized[i]['Close'] = normalized_data[i][3]
    allFilesMergedChunks_normalized[i]['Volume'] = normalized_data[i][4]
    allFilesMergedChunks_normalized[i].reset_index(drop=True, inplace=True)



"""
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
"""
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
whitesoldiers = []
starsinsouth = []
twocrows = []
linearreg = []
stddev = []
tsf = []
"""
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
"""
# Calculate different parameters from different TA-LIB classes
for chunkDf in allFilesMergedChunks:
        # CLASS_1 : 'Overlap Studies' - Bollinger Bands (with Simple Moving Average)
        upper, middle, lower = TaLib_Calculations.getBBands(chunkDf, chunkSize)
        upper_df = pandas.DataFrame(data = upper)
        middle_df = pandas.DataFrame(data = middle)
        lower_df = pandas.DataFrame(data = lower)
        upperBB.append(upper_df)
        middleBB.append(middle_df)
        lowerBB.append(lower_df)
        #CLASS_1 : 'Overlap Studies' - Midpoint over Period
        midpoint.append(pandas.DataFrame(data = TaLib_Calculations.getMidpoint(chunkDf, chunkSize)))
        #CLASS_1 : 'Overlap Studies' - Weighted Moving Average
        wma.append(pandas.DataFrame(data =TaLib_Calculations.getWMA(chunkDf, chunkSize)))
        # CLASS_2 : 'Momentum Indicators' - Momentum
        mom.append(pandas.DataFrame(data =TaLib_Calculations.getMomentum(chunkDf, chunkSize)))
        # CLASS_2 : 'Momentum Indicators' - Money Flow Index
        mfi.append(pandas.DataFrame(data =TaLib_Calculations.getMFI(chunkDf, chunkSize)))
        # CLASS_2 : 'Momentum Indicators' - Balance of Power
        bop.append(pandas.DataFrame(data =TaLib_Calculations.getBOP(chunkDf)))
        # CLASS_3 : 'Volume Indicators' - Chaikin Accumulation/Distribution Line
        adline.append(pandas.DataFrame(data =TaLib_Calculations.getADLine(chunkDf)))
        # CLASS_3 : 'Volume Indicators' - Chaikin A/D Oscillator
        adosc.append(pandas.DataFrame(data =TaLib_Calculations.getADOscillator(chunkDf, chunkSize)))
        # CLASS_3 : 'Volume Indicators' - On Balance Volume
        obv.append(pandas.DataFrame(data =TaLib_Calculations.getOBV(chunkDf)))
        # CLASS_4 : 'Price Transform' - Average Price
        avgprice.append(pandas.DataFrame(data =TaLib_Calculations.getAvgPrice(chunkDf)))
        # CLASS_4 : 'Price Transform' - Typical Price
        typprice.append(pandas.DataFrame(data =TaLib_Calculations.getTypicalPrice(chunkDf)))
        # CLASS_4 : 'Price Transform' - Weighted Close Price
        wclprice.append(pandas.DataFrame(data =TaLib_Calculations.getWClPrice(chunkDf)))
        # CLASS_5 : 'Volatility Indicators - Average True Range
        atr.append(pandas.DataFrame(data =TaLib_Calculations.getAverageTrueRange(chunkDf, chunkSize)))
        # CLASS_5 : 'Volatility Indicators - Normalized Average True Range
        natr.append(pandas.DataFrame(data =TaLib_Calculations.getNATR(chunkDf, chunkSize)))
        # CLASS_5 : 'Volatility Indicators - True Range (NaN an erster Stelle rausfilern?)
        tr.append(pandas.DataFrame(data = TaLib_Calculations.getTrueRange(chunkDf)))
        # CLASS_6 : 'Pattern Recognition' - Three Advanced White Soldiers
        whitesoldiers.append(pandas.DataFrame(data =TaLib_Calculations.get3AWS(chunkDf)))
        # CLASS_6 : 'Pattern Recognition' - Three Stars in the South
        starsinsouth.append(pandas.DataFrame(data =TaLib_Calculations.get3SITS(chunkDf)))
        # CLASS_6 : 'Pattern Recognition' - Two Crows
        twocrows.append(pandas.DataFrame(data =TaLib_Calculations.get2Crows(chunkDf)))
        # CLASS_7 : 'Statistic Functions' - Linear Regression
        linearreg.append(pandas.DataFrame(data =TaLib_Calculations.getLinearReg(chunkDf, chunkSize)))
        # CLASS_7 : 'Statistic Functions' - Standard Deviation
        stddev.append(pandas.DataFrame(data =TaLib_Calculations.getStdDev(chunkDf, chunkSize)))
        # CLASS_7 : 'Statistic Functions' - Time Series Forecast
        tsf.append(pandas.DataFrame(data =TaLib_Calculations.getTSF(chunkDf, chunkSize)))


print("TALIB DATA CALCULATION finished ...")

# Normalize Talib Data
columns = numpy.array(['upperBB','middleBB','lowerBB','midpoint','wma','mom','mfi','bop','adline',
           'adosc','obv','atr','natr','tr','avgprice','typprice','wclprice',
           'whitesoldiers','starsinsouth','twocrows','linearreg','stddev','tsf'])

TaLib_list = [upperBB, middleBB, lowerBB, midpoint, wma, mom, mfi, bop, adline, adosc, obv, atr, natr, tr, avgprice,
                          typprice, wclprice, whitesoldiers, starsinsouth, twocrows, linearreg, stddev, tsf]

"""
#merge all TaLib Data for all Files
Talib_merged = []
for list in TaLib_list:
    Talib_merged.append(pandas.concat(list))

#normalize all TaLib Data for all Files
Talib_merged_normalized = []
for dataFrame in Talib_merged:
    min_max_scaler = preprocessing.MinMaxScaler()
    x = dataFrame.values
    x_scaled = min_max_scaler.fit_transform(x)
    table = pandas.DataFrame(x_scaled)
    Talib_merged_normalized.append(table)
"""

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

"""
#Mesrge normalized Chunks and Combine with Ta-Lib Data
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
"""
columns = numpy.array(['Date','Open','High','Low','Close','Volume','Ticker',
                       'upperBB','middleBB','lowerBB','midpoint','wma','mom','mfi','bop','adline',
                       'adosc','obv','atr','natr','tr','avgprice','typprice','wclprice',
                       'whitesoldiers','starsinsouth','twocrows','linearreg','stddev','tsf'])
"""
mergedBaseData = pandas.concat(ChunksMergedList_validated)
concat_data = [mergedBaseData]
for df in Talib_merged_normalized:
    concat_data.append(df)
allFilesAllData = pandas.concat(concat_data, axis=1, join="inner")
allFilesAllData.columns = columns
allFilesAllDataNoNaNs = allFilesAllData.dropna()

dataFramesPickleSize = []
df_split = numpy.array_split(allFilesAllDataNoNaNs, numberOfFilesToRead)
for dataFrames in df_split:
    temp = numpy.array_split(dataFrames, (chunksPerFileToRead-2))
    for df in temp:
        if len(df.index) > chunkSize:
            df.drop(df.tail(1).index, inplace=True)
    dataFramesPickleSize.append(temp)
print(dataFramesPickleSize)
"""
i = 0
allFilesAllData = []
while i < len(allFilesMergedChunks_normalized):
    allFilesAllData.append(pandas.concat([allFilesMergedChunks_normalized[i], upperBB[i], middleBB[i], lowerBB[i], midpoint[i],
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

dataFramesPickleSize = []
for dataFrames in allFilesAllDataNoNans:
    temp = numpy.array_split(dataFrames, (chunksPerFileToRead-2))
    for df in temp:
        if len(df.index) > chunkSize:
            df.drop(df.tail(1).index, inplace=True)
    dataFramesPickleSize.append(temp)


# Save as pickle file
for dataFramesList in dataFramesPickleSize:
    i = 0
    for dataFrames in dataFramesList:
        if (dataFrames.empty):
            print("leer")
        else:
            i = i + 1
            print(str(dataFrames.iloc[1]['Ticker']))
            dataFrames.to_pickle("./Pickle/" + dataFrames.iloc[1]['Ticker'] + "_" + str(i) + ".pkl")
print("PICKLE FILE CREATION finished... ~" + str(i*numberOfFilesToRead) + " Files persistiert.")
