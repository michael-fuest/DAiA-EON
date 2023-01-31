import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk
from darts import utils
from darts.models import RegressionModel
from darts.models import BlockRNNModel
from darts.models import NaiveSeasonal
from darts import TimeSeries
from darts.metrics import mape
from darts.metrics import rmse

#path to the data folder, edit if necesssary
path = "./data/Generated Data/"

#Price Files 
gas_prices = 'gas_prices.csv'
power_prices = 'power_prices.csv'

#Master files
master_power = 'master_competitor_market_prices_power.csv'
master_gas = 'master_competitor_market_prices_gas.csv'

#Strategy Index aggregated files
power_strat = 'Strat_competitor_market_prices_power.csv'
gas_strat = 'Strat_competitor_market_prices_gas.csv'

#Reading in relevant datasets
df_power = pd.read_csv(path + master_power)
df_gas = pd.read_csv(path + master_gas)
df_power_strat = pd.read_csv(path + power_strat)
df_gas_strat = pd.read_csv(path + gas_strat)
df_power_prices = pd.read_csv(path + power_prices)
df_gas_prices = pd.read_csv(path + gas_prices)


def main(postcode, consumption_range, rank_range, household_data, price_data, days):
    provideForecasts(postcode, consumption_range, rank_range, household_data, price_data, days)


def eval_linear_model(model, target, covariates, days = 1, train_size = 0.8):

    #Defining test dataset
    p_train, _ = target.split_before(train_size)
    model.fit(p_train, past_covariates= covariates)
        
    # We backtest the model on the last part of the timeseries:
    backtest = model.historical_forecasts(series=target, 
                                            past_covariates=covariates,
                                            start=train_size, 
                                            retrain=True,
                                            verbose=False, 
                                            forecast_horizon=days)
    return rmse(target, backtest)


def evaluate_models(target, covariates, lags, lags_covs, days = 1, train_size = 0.80):
    days = max(days, 1)
    best_score, best_cfg = float("inf"), None
    for l in lags:
        for c in lags_covs:
            try:
                reg_model = RegressionModel(lags = l, lags_past_covariates= c, output_chunk_length=days)
                mse = eval_linear_model(reg_model, target, covariates, days, train_size)
                if mse < best_score:
                    best_score, best_cfg = mse, (l, c)
            except:
                print('error in model eval')
                continue
    print('Best ModelA%s MSE=%.6f' % (best_cfg, best_score))
    return best_cfg


def querySingleTariffRange(rank = (1,5), post_code = 81737, consumption_range = 2500, df_power = df_power) -> pd.DataFrame:

    res = df_power.loc[(df_power['rank'] >= rank[0]) & (df_power['rank'] <= rank[1]) & (df_power.post_code == post_code) & (df_power.consumption_range_kwh == consumption_range)].copy()
    if res.shape[0] > 0:
        #Generating date indexed data
        res["valid_range"] = res.apply(lambda x: pd.date_range(x["date_valid_from"], x["date_valid_to"]), axis=1)
        res = res.explode("valid_range").copy()

        #Aggregating for mean household prices per day
        res = res.groupby('valid_range').agg({'price_kwh':'mean'})
    return res    

def provideForecasts(postcode, consumption_range, rank_range, household_data, price_data, days):

    #Querying dataset
    df = querySingleTariffRange(rank_range, postcode, consumption_range, household_data)
    
    if not df.shape[0]:
        print('No data found.')
        return

    #Shifting prices by 1 to avoid look-ahead bias
    real_prices = df.price_kwh
    df = df.shift(1)
    df.price_kwh = real_prices

    #Merging price data
    df = df.join(price_data)

    #Adding rolling moving average as additional covariate
    df['moving_average'] = df.avg_price.ewm(alpha=0.1, adjust=False).mean()

    #Dropping resulting NA column
    df.dropna(inplace=True)
    
    ##Building TimeSeries objects, and filling in missing date indices
    past_covs = utils.missing_values.fill_missing_values(TimeSeries.from_dataframe(df[['avg_price', 'moving_average']], fill_missing_dates= True))
    prices = utils.missing_values.fill_missing_values(TimeSeries.from_dataframe(df[['price_kwh']], fill_missing_dates= True))

    #Defining train datasets
    past_covs_train = past_covs[:-days]
    prices_train = prices[:-days]
    prices_valid = prices[-days:]

    #Defining hyperparam grid for linear regression model
    lags = [1,2,3,4,5]
    cov_lags = [1,2,3,4,5]

    #Finding optimal lags
    l, c = evaluate_models(prices_train, past_covs_train, lags, cov_lags, days)

    #Defining Benchmark
    benchmark = NaiveSeasonal(K=1)
    benchmark.fit(prices_train)
    bm = benchmark.predict(days)


    reg_model = RegressionModel(l, c, output_chunk_length=days)
    reg_model.fit(prices_train, past_covs_train)
    rnn_model = BlockRNNModel(input_chunk_length=days, output_chunk_length=days, n_rnn_layers=2, random_state = 42)
    rnn_model.fit(prices_train, past_covariates=past_covs_train, epochs=200, verbose = False)
    preds_rnn = rnn_model.predict(days)
    preds_reg = reg_model.predict(days)


    prices[-30:].plot(label= 'timeseries')
    preds_rnn.plot(label = 'predictions RNN')
    preds_reg.plot(label = 'predictions Regression')
    bm.plot(label = 'Naive benchmark')

    print('Baseline MAPE: ', mape(prices[-days:], bm))
    print('RNN MAPE: ', mape(prices[-days:], preds_rnn))
    print('Reg MAPE: ', mape(prices[-days:], preds_reg))

    res = pd.DataFrame({'actual_timeseries': pd.Series(prices_valid), 'Linear Regression Forecast': pd.Series(preds_reg), 'RNN Forecast': pd.Series(preds_rnn)})
    return res

    
if __name__ == "__main__":
    main()

