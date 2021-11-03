import pandas
import pickle
from sklearn import preprocessing

data = pandas.read_csv('c:/data/datasets/stocks/aki_subset/stocks.csv')

time = min(data.Date)
end = max(data.Date)
#end = 253 nur zum testen

T = 20

# Für jeden einzelnen Zeitpunkt
while time < end - T:
    print(time)
    # Für alle Firmen die zum Zeitpunkt 0 existent waren -> erste 20 Einträge rausgefiltert
    subset = data[(data['Date'] >= time) & (data['Date'] < time + T + 1)]
    # Tabelle wird gedreht
    table = pandas.pivot_table(subset, index=['Date'], columns=['Ticker'])
    table.columns = table.columns.get_level_values(1)
    # Fehlende Werte killen
    table = table.dropna(axis=1)
    # Alle Aktien die keine Variation haben (immer gleichen Wert über 20 Tage) -> rauswerfen
    table = table.loc[:,table.nunique()!=1]

    #normalize
    x = table.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    table = pandas.DataFrame(x_scaled, columns=table.columns)

    # Ziele
    targets = table.iloc[T]
    # Rest ausser Ziel wird im Table gehalten
    table = table[0:T]
    # Korrelationen werden berechnet -> zum Zeitpunkt 0 ... (steigt je loop) pro einzelne Aktie
    correlations = table.corr()
    correlations = correlations.dropna(axis=1)

    # Wir gehen jetzt durch die Spalten
    for col in correlations.columns:
        item = dict()
        item['y'] = targets[col]
        # Alle Y Werte als input
        item['Y'] = table[col]
        pos_stocks = list(correlations[col].nlargest(21).index) # largest correlation is with stock itself
        pos_stocks.remove(col)
        item['X_p'] = table[pos_stocks]
        neg_stocks = list(correlations[col].nsmallest(20).index)
        item['X_n'] = table[neg_stocks]
        with open('c:/data/htw/2021_SS/AKI/Samples/' + col + '_' + str(time) + '.pkl', 'wb') as f:
            pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
    time = time + 1