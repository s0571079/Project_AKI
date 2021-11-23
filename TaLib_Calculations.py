import talib
import numpy

# Bollinger Bands
def getBBands(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None, None, None

    try:
        upper, middle, lower = talib.BBANDS(close.values,
                                            timeperiod=45,
                                            # number of non-biased standard deviations from the mean
                                            nbdevup=1,
                                            nbdevdn=1,
                                            # Moving average type: simple moving average here
                                            matype=0)
    except Exception as ex:
        return None, None, None

    return upper, middle, lower

#Midpoint over Period
def getMidpoint(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        midpoint = talib.MIDPOINT(close.values,
                                  timeperiod=45)
    except Exception as ex:
        return None

    return midpoint

#Momentum
def getMomentum(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        mom = talib.MOM(close.values,
                        timeperiod=44)
    except Exception as ex:
        return None

    return mom

#Chaikin Accumulation/Distribution Line
def getADLine(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume'].astype(float)
    except Exception as ex:
        return None

    try:
        adline = talib.AD(high.values,
                          low.values,
                          close.values,
                          volume.values)
    except Exception as ex:
        return None

    return adline

#Average True Range
def getAverageTrueRange(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        atr = talib.ATR(high.values,
                        low.values,
                        close.values,
                        timeperiod=44)
    except Exception as ex:
        return None

    return atr

#Average Price
def getAvgPrice(df):
    try:
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        avgprice = talib.AVGPRICE(open.values,
                          high.values,
                          low.values,
                          close.values)
    except Exception as ex:
        return None

    return avgprice

#Hilbert Transform - Dominant Cycle Period
def getHTDCPeriod(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        htdcperiod = talib.HT_DCPERIOD(close.values)
    except Exception as ex:
        return None

    return htdcperiod

#Hilbert Transform - Dominant Cycle Phase
def getHTDCPhase(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        htdcphase = talib.HT_DCPHASE(close.values)
    except Exception as ex:
        return None

    return htdcphase

#Three Advancing White Soldiers
def get3AWS(df):
    try:
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        whitesoldiers = talib.CDL3WHITESOLDIERS(open.values,
                                                high.values,
                                                low.values,
                                                close.values)
    except Exception as ex:
        return None

    return whitesoldiers

#Three Advancing White Soldiers
def get3SITS(df):
    try:
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        starsinsouth = talib.CDL3STARSINSOUTH(open.values,
                                              high.values,
                                              low.values,
                                              close.values)
    except Exception as ex:
        return None

    return starsinsouth

#Two Crows
def get2Crows(df):
    try:
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        twocrows = talib.CDL2CROWS(open.values,
                                   high.values,
                                   low.values,
                                   close.values)
    except Exception as ex:
        return None

    return twocrows

#Linear Regression
def getLinearReg(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        linearreg = talib.LINEARREG(close.values,
                                    timeperiod=45)
    except Exception as ex:
        return None

    return linearreg

#Money Flow Index
def getMFI(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume'].astype(float)
    except Exception as ex:
        return None

    try:
        mfi = talib.MFI(high.values,
                        low.values,
                        close.values,
                        volume.values,
                        timeperiod=44)
    except Exception as ex:
        return None

    return mfi

#Weighted Moving Average
def getWMA(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        wma = talib.WMA(close.values,
                        timeperiod=45)
    except Exception as ex:
        return None

    return wma

#Balance of Power
def getBOP(df):
    try:
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        bop = talib.BOP(open.values,
                        high.values,
                        low.values,
                        close.values)
    except Exception as ex:
        return None

    return bop

#Chaikin A/D Oscillator
def getADOscillator(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume'].astype(float)
    except Exception as ex:
        return None

    try:
        adosc = talib.ADOSC(high.values,
                            low.values,
                            close.values,
                            volume.values,
                            fastperiod=3,
                            slowperiod=44)
    except Exception as ex:
        return None

    return adosc

#On Balance Volume
def getOBV(df):
    try:
        close = df['Close']
        volume = df['Volume'].astype(float)
    except Exception as ex:
        return None

    try:
        obv = talib.OBV(close.values,
                        volume.values)
    except Exception as ex:
        return None

    return obv

#Typical Price
def getTypicalPrice(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        typprice = talib.TYPPRICE(high.values,
                                  low.values,
                                  close.values)
    except Exception as ex:
        return None

    return typprice

#Weighted Close Price
def getWClPrice(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        wclprice = talib.WCLPRICE(high.values,
                                 low.values,
                                 close.values)
    except Exception as ex:
        return None

    return wclprice

#Normalized Average True Range
def getNATR(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        natr = talib.ATR(high.values,
                        low.values,
                        close.values,
                        timeperiod=44)
    except Exception as ex:
        return None

    return natr

#True Range
def getTrueRange(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
    except Exception as ex:
        return None

    try:
        tr = talib.TRANGE(high.values,
                          low.values,
                          close.values)
    except Exception as ex:
        return None

    return tr

#Standard Deviation
def getStdDev(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        stddev = talib.STDDEV(close.values,
                              timeperiod=45)
    except Exception as ex:
        return None

    return stddev

#Time Series Forecast
def getTSF(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        tsf = talib.TSF(close.values,
                        timeperiod=45)
    except Exception as ex:
        return None

    return tsf