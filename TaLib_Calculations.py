import talib

# Bollinger Bands
def getBBands(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None, None, None

    try:
        upper, middle, lower = talib.BBANDS(close.values,
                                            timeperiod=len(close),
                                            # number of non-biased standard deviations from the mean
                                            nbdevup=1,
                                            nbdevdn=1,
                                            # Moving average type: simple moving average here
                                            matype=0)
    except Exception as ex:
        return None, None, None

    return upper, middle, lower

#Double Exponential Moving Average - timeperiod>11 = nur noch NaN
def getDEMA(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        dema = talib.DEMA(close.values,
                          timeperiod=10)
    except Exception as ex:
        return None

    return dema

#Momentum
def getMomentum(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None

    try:
        mom = talib.MOM(close.values,
                        timeperiod=len(close)-1)
    except Exception as ex:
        return None

    return mom

#Chaikin Accumulation/Distribution Line
def getADLine(df):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
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
                        timeperiod=len(close)-1)
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
                                    timeperiod=len(close))
    except Exception as ex:
        return None

    return linearreg