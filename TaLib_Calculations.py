import talib

def getBBands(df):
    try:
        close = df['Close']
    except Exception as ex:
        return None, None, None

    try:
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=10,
            # number of non-biased standard deviations from the mean
            nbdevup=1,
            nbdevdn=1,
            # Moving average type: simple moving average here
            matype=0)
    except Exception as ex:
        return None, None, None

    return upper, middle, lower