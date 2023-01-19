import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

path = 'D:/DAiA/Eon/Master data/output/DAiA-EON/data/'
ll_data = 'leaf_level/group/'
csv = '.csv'

mkt_data = pd.read_csv(path+'power_market_MA'+csv, parse_dates = ["date"])
mkt_data.sort_values("date", inplace=True)
mkt_data.drop('Unnamed: 0', axis=1, inplace=True)

files = pd.read_csv(path+'segregated_supplier_file_list'+csv)
files.drop(columns = ['Unnamed: 0'], inplace = True)
files.rename(columns = {"0" : "File_name"}, inplace=True)
data_files = files['File_name'].to_list()
#data_files
data_files = data_files[0:100000]
#data_files[1:10]

RF_results = []
RF_results1 = []

def R_forest(data_file):
    supp_data = pd.read_csv(path+ll_data+data_file, parse_dates = ["date"])
    supp_data.sort_values("date", inplace=True)
    
    if supp_data.shape[0]>100: 
        supp_data = supp_data.groupby('date').agg({'price_year_eur':'mean'})

        supp_data = supp_data.merge(mkt_data, how='inner', on='date')

        supp_data.corr()

        new_mkt = supp_data.copy()
        new_mkt.drop(columns = ['price_year_eur'], inplace = True)
        new_mkt.set_index('date', inplace=True)

        X= new_mkt
        X

        
        y = supp_data['price_year_eur']
    
        # Using PCA to find the most relevant price instruments
        pca = PCA(0.99)
        X_pca = pca.fit_transform(X)
        num_ins = pca.n_components_
        
    
        # Split the data into training and test sets
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
        # Train a random forest model
        model = RandomForestRegressor(n_estimators=150, max_depth = 20, max_features='auto')
        model.fit(X_train_pca, y_train)
    
        # Test the model on the test data
        predictions = model.predict(X_test_pca)
    
        # Evaluate the model's performance
        # Mean absolute error
        mae = np.mean(abs(predictions - y_test))
        print(f'Mean absolute error: {mae:.2f}')
        # R^2
        r = r2_score(y_test, predictions)
        print('R squared:', r)
        
        # Use the model to make predictions on new data
        #new_energy_market_prices = [[210, 150, 165, 200, 185], [240, 180, 195, 230, 220], [230, 170, 185, 220, 210]]
        #predicted_supplier_prices = model.predict(new_energy_market_prices)
        #print(f'Predicted supplier prices: {predicted_supplier_prices}')
    
        # Calaulating model accuracy
        acc = model.score(X_test_pca, y_test)
    
        #df1 = supp_data.groupby('date').agg({'price_year_eur':'mean'})
        #df1.join(X_test_pca).plot()
        
        imp = model.feature_importances_# Importance factor of features
        ind = imp.argsort()[-num_ins:][::-1] # Indices of most significant features
        feature = X.columns
        sig_features = [feature[i] for i in ind] # Get the most significant features
        sig_features_imp = imp[ind] # Importance of significant features
       
        for j in range(num_ins):
            RF_supp_data1 = {'File': data_file,'R sq': r,'Score': acc, 'MAE':mae,'Reg_ins':sig_features[j], 'Reg_coeff': sig_features_imp[j], 'No. of relevant instruments': num_ins}
            RF_results1.append(RF_supp_data1)
        
        
        #Get the 1st PC's coefficients for relevant dimensions
        pc1_coeff = pca.components_[0]
        
        ins = new_mkt.columns
        j=0
        # Find the most significant dimensions & their explanatory powers
        pca_dim = sorted(zip(ins, pc1_coeff), key=lambda x: abs(x[1]), reverse=True)[:num_ins]
        for dim ,coeff in pca_dim:
            RF_supp_data = {'File': data_file,'R sq': r,'Score': acc, 'MAE':mae,'Reg_ins':sig_features[j], 'Reg_coeff': sig_features_imp[j], 'PCA_ins':dim, 'PCA_Variance': coeff, 'No. of relevant instruments': num_ins}
            RF_results.append(RF_supp_data)
        print(str(data_file))
        
        
    else:
        return pd.DataFrame()
    
    return acc

for data_file in data_files:
    R_forest(data_file)
    print(data_files.index(data_file))
    
df = pd.DataFrame(RF_results)    
df.to_csv('D:/DAiA/Eon/Master data/output/DAiA-EON/data/RF_PCA_results.csv')

df1 = pd.DataFrame(RF_results1)    
df1.to_csv('D:/DAiA/Eon/Master data/output/DAiA-EON/data/RF_Reg_coeff_results.csv')
