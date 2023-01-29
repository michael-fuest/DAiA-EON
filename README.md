# What is the best pricing strategy for XXX's household pricing?

## Project Description ##

In times of high energy market price volatility, caused by Russia's invasion of Ukraine, energy suppliers who need to source energy themselves are trying to determine the optimal pricing strategy for private household energy contracts. Using public energy market data, and competitor data from the top 100 postcodes in Germany, we investigate different household pricing reactions to energy market price developments, and model household energy prices as a function of their own lagged values, lagged energy market prices and exponential moving averages of energy market prices. We use libraries like Pandas, for data wrangling and pre-processing, plotly for creating interactive visualizations and darts for timeseries modeling and forecasting. Project results were deployed in a static Tableau dashboard, visualizing descriptive analyses and model forecasts for exemplary postcode consumption range combinations.

## How to run and install ##

We use virtual environments to run the included notebooks. Other environment managers like conda can also be used. Installing jupyter (and maybe ipykernel) as well as all of the listed packages is sufficient to get this code to run:

-Pandas
-Numpy
-Matplotlib
-Plotly
-Darts
-Seaborn
-Scikit-learn
-(statsmodels)
-(jupyter)

## Contents ##

This project contains three dedicated notebooks:

-> Master Dataset.ipynb
Notebook for data aggregation, loading, cleaning and feature engineering.

-> Exploration.ipynb
Notebook for data exploration, visualization and descriptive analysis for price movements and margins

-> Timeseries.ipynb
Notebook for timeseries forecasting experiments, which outputs aggregated predictions for around 900 postcodes for both power and gas pricing data.


## Credits ##

This project was conducted at the Technical University of Munich (TUM) Forschungsinstitut – Unternehmensführung, Logistik und Produktion, in cooperation with Sebastian Junker and the project team consisting of Richard Jasansky (https://github.com/RichardJasansky), Chaitanya Shinde (https://github.com/CS-5627), Dominik Schäfer (https://github.com/dominik-99) and Nick Knappstein (https://github.com/NickKnappstein)
