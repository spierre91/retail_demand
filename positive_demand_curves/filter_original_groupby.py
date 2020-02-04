import pandas as pd  

df = pd.read_csv("upc_gt5_unique_discounts.csv")
print(df.head())
rsquare = pd.read_csv("upc_dicsount_sppd_r2_gt08_update.csv")
print(rsquare.head())



df = df[df['upc'].isin(list(rsquare['upc'].values))]
df.to_csv("UPC_paramater_combo.csv")
print(df.head(100))
