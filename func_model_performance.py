import numpy as np
import pandas as pd
import tensorflow as tf
from darts.metrics import mape, mse
from darts.models import RegressionModel, BlockRNNModel, NaiveSeasonal
from darts import utils, TimeSeries
from darts.dataprocessing.transformers import Scaler
from window_generator import WindowGenerator

tf.random.set_seed(42)

def queryRangePostCode(df_power, consumption_range = 2500, post_code = 81737, rank_range = (1,5)) -> pd.DataFrame:

    res = df_power.loc[(df_power['rank'] >= rank_range[0]) & (df_power['rank'] <= rank_range[1]) & (df_power.post_code == post_code) & (df_power.consumption_range_kwh == consumption_range)].copy()
    if res.shape[0] > 0:
        #Generating date indexed data
        res["valid_range"] = res.apply(lambda x: pd.date_range(x["date_valid_from"], x["date_valid_to"]), axis=1)
        res = res.explode("valid_range").copy()

        #Aggregating for mean household prices per day
        res = res.groupby('valid_range').agg({'price_kwh':'mean'})
    return res

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

error_metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]

def compile_and_fit(model, window, error_metrics = error_metrics, patience=2, max_epochs = 10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    verbose = 0)

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=error_metrics)

  
  history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=early_stopping, shuffle = True, verbose = 0)
  return history



def test_tensorflow_impl(shift, consRanges, postCodes, household_data, market_data, rank_range = (6,10), input_width = 10):
    
    #columns = pd.MultiIndex.from_product([["baseline","dense","convolutional", "LSTM" ], [errormetric.name for errormetric in error_metrics]])
    models = ["baseline","dense","convolutional", "LSTM" ]
    mse_df = pd.DataFrame(index = models)
    mape_df = pd.DataFrame(index = models)
    
    for postCode in postCodes:
        for consRange in consRanges:
            
            print(postCode, consRange, end = "; ")
            
            #Data preparation and normalization
            try:
                norm_layer = tf.keras.layers.Normalization(axis = 1)
                input = queryRangePostCode(household_data, consumption_range=consRange, post_code=postCode, rank_range=rank_range)
                input = input.join(market_data[["avg_price", "moving_average"]]).interpolate(method = "time", axis = 0).dropna(axis = 0)
                norm_layer.adapt(input)
                norm_input = norm_layer(input)
                norm_input = pd.DataFrame(norm_input, columns = input.columns, index = input.index)
                train_df = norm_input[:-shift]
                val_df = norm_input[-(2*shift+input_width):]
            except:
                continue


            #Generate Windows
            single_step_window = WindowGenerator(input_width=1, label_width=1, shift=shift,
                                        label_columns=['price_kwh'], train_df=train_df, val_df=val_df)

            multi_in_single_out = WindowGenerator(input_width=input_width, label_width=1, shift=shift,
                                        label_columns=['price_kwh'], train_df=train_df, val_df=val_df)


            #Clearing of previous parameters
            tf.keras.backend.clear_session()


            #Model Declaration
            #baseline
            baseline = Baseline(label_index = 0)

            #Dense NN
            dense = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='sigmoid'),
                tf.keras.layers.Dense(units=64, activation='sigmoid'),
                tf.keras.layers.Dense(units=1),
                tf.keras.layers.Reshape([1, -1]),
            ])

            #Convolutional NN with 1 Dense Layer
            conv_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32, kernel_size=(input_width,), activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1),
            ])

            #Simple LSTM Model
            lstm_model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(units=1)
            ])

            #create a dict for model loop and pd.Series for storing MSE and MAPE
            models = {"dense": dense, "convolutional": conv_model, "LSTM": lstm_model}

            mse_s = pd.Series(name = (postCode, consRange), dtype = np.float64)
            mape_s = pd.Series(name = (postCode, consRange), dtype = np.float64)

            
            #Fit Models
            baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=error_metrics)
            
            mse_s["baseline"], mape_s["baseline"] = baseline.evaluate(single_step_window.val, verbose = 0)
            
            
            for name, model in models.items():
                history = compile_and_fit(model, multi_in_single_out)
                mse_s[name], mape_s[name]  = model.evaluate(multi_in_single_out.val, verbose = 0)

            
            #Store MSE and MAPE Values in DFs
            mse_df = pd.concat([mse_df, mse_s], axis = 1)
            mape_df = pd.concat([mape_df, mape_s], axis = 1)

    return mse_df.T, mape_df.T

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
    return mape(target, backtest)


def evaluate_models(target, covariates, lags, lags_covs, days = 1, train_size = 0.80):
    days = max(days, 1)
    best_score, best_cfg = float("inf"), None
    for l in lags:
        for c in lags_covs:
            try:
                reg_model = RegressionModel(lags = l, lags_past_covariates= c, output_chunk_length=days)
                mape = eval_linear_model(reg_model, target, covariates, days, train_size)
                if mape < best_score:
                    best_score, best_cfg = mape, (l, c)
            except:
                print('error in model eval')
                continue
    print('Best ModelA%s MAPE=%.6f' % (best_cfg, best_score))
    return best_cfg


def scaleTimeSeries(timeseries):
    scaler = Scaler()
    series_scaled = scaler.fit_transform(timeseries)
    return series_scaled


def test_darts_impl(shift, consRanges, postCodes, household_data, market_data, rank_range = (6,10)):

    models = ["baseline","LSTM","regression"]
    mse_df = pd.DataFrame(index = models)
    mape_df = pd.DataFrame(index = models)

    for postCode in postCodes:
        for consRange in consRanges:
            
            print(postCode, consRange, end = "; ")

            #Querying dataset
            df = queryRangePostCode(rank_range = rank_range, post_code=postCode, consumption_range=consRange, df_power=household_data)
            
            if not df.shape[0]:
                print('No data found.')
                continue

            #Shifting prices by 1 to avoid look-ahead bias
            real_prices = df.price_kwh
            df = df.shift(1)
            df.price_kwh = real_prices

            #Merging price data
            df = df.join(market_data)

            #Adding rolling moving average as additional covariate
            df['moving_average'] = df.avg_price.ewm(alpha=0.1, adjust=False).mean()

            #Dropping resulting NA column
            df.dropna(inplace=True)
            
            ##Building TimeSeries objects, and filling in missing date indices
            past_covs = utils.missing_values.fill_missing_values(TimeSeries.from_dataframe(df[['avg_price', 'moving_average']], fill_missing_dates= True))
            prices = utils.missing_values.fill_missing_values(TimeSeries.from_dataframe(df[['price_kwh']], fill_missing_dates= True))

            #Scaling both timeseries
            past_covs = scaleTimeSeries(past_covs)
            prices = scaleTimeSeries(prices)

            #Defining train datasets
            past_covs_train = past_covs[:-shift]
            prices_train = prices[:-shift]
            prices_valid = prices[-shift:]

            #Defining hyperparam grid for linear regression model
            lags = [1,2,3,4,5]
            cov_lags = [1,2,3,4,5]

            #Finding optimal lags
            l, c = evaluate_models(prices_train, past_covs_train, lags, cov_lags, shift)

            #Defining Benchmark
            benchmark = NaiveSeasonal(K=1)
            benchmark.fit(prices_train)
            bm = benchmark.predict(shift)


            #Fit and predict Models 
            reg_model = RegressionModel(l, c, output_chunk_length=shift)
            reg_model.fit(prices_train, past_covs_train)
            rnn_model = BlockRNNModel(model="LSTM", n_epochs=20, random_state=42, input_chunk_length= 1, output_chunk_length=shift)
            rnn_model.fit(prices_train, past_covariates=past_covs_train, verbose = False)
            preds_rnn = rnn_model.predict(shift, series = prices_train, past_covariates= past_covs_train)
            preds_reg = reg_model.predict(shift, series = prices_train, past_covariates= past_covs_train)


            mse_s = pd.Series(name = (postCode, consRange), dtype = np.float64)
            mape_s = pd.Series(name = (postCode, consRange), dtype = np.float64)


            mape_s["LSTM"] = mape(preds_rnn, prices_valid)
            mape_s["regression"] = mape(preds_reg, prices_valid)
            mape_s["baseline"] = mape(bm, prices_valid)

            mse_s["LSTM"] = mse(preds_rnn, prices_valid)
            mse_s["regression"] = mse(preds_reg, prices_valid)
            mse_s["baseline"] = mse(bm, prices_valid)

            #Store MSE and MAPE Values in DFs
            mse_df = pd.concat([mse_df, mse_s], axis = 1)
            mape_df = pd.concat([mape_df, mape_s], axis = 1)

    
    return mse_df.T, mape_df.T