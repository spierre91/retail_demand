import pandas as pd
import matplotlib.pyplot as plt 


commands = pd.read_json("commands.json",  lines=True)
out = pd.read_csv("output.csv")
out = out.iloc[2:64901]
print(out.head())
discount_perc = []
actual = []
Upc = []
predict = []
for i in list(commands['args'].values):
    discount_perc.append(i['discount_perc'])
    
for i in list(commands['args'].values):
    Upc.append(i['Upc'])

for j in list(commands['actual'].values):
    actual.append(j)
    
for k in list(out['predicted'].values):
    predict.append(k)
    
df = pd.DataFrame({"discount_perc": discount_perc, "actual": actual, 'Upc': Upc})

df = df.head(len(out))




df['predict'] = predict
#df = df[df["Upc"] == '08-51107-00305']

#print(df.head())
#print(set(Upc))
#plt.scatter(df["discount_perc"], df["actual"])
#plt.scatter(df["discount_perc"], df["predict"])

df_zero = df[df['discount_perc'] == 0]


df_005 = df[df['discount_perc'] >= 0.05]
df_005 = df_005[df_005['discount_perc'] < 0.1]



df_01 = df[df['discount_perc'] >= 0.1]
df_01 = df_01[df_01['discount_perc'] < 0.15]

df_015 = df[df['discount_perc'] >= 0.15]
df_015 = df_015[df_015['discount_perc'] < 0.2]

df_02 = df[df['discount_perc'] >= 0.2]
df_02 = df_02[df_02['discount_perc'] < 0.25]


df_025 = df[df['discount_perc'] >= 0.25]
df_025 = df_025[df_025['discount_perc'] < 0.3]


df_03 = df[df['discount_perc'] >= 0.3]
df_03 = df_03[df_03['discount_perc'] < 0.35]

df_035 = df[df['discount_perc'] >= 0.35]
df_035 = df_035[df_035['discount_perc'] < 0.4]



df_04 = df[df['discount_perc'] >= 0.4]
df_04 = df_04[df_04['discount_perc'] < 0.45]


df_045 = df[df['discount_perc'] >= 0.45]
df_045 = df_045[df_045['discount_perc'] < 0.5]


df_05 = df[df['discount_perc'] >= 0.5]
df_05 = df_05[df_05['discount_perc'] < 0.55]

df_055 = df[df['discount_perc'] >= 0.55]
df_055 = df_055[df_055['discount_perc'] < 0.6]


df_06 = df[df['discount_perc'] >= 0.6]
df_06 = df_06[df_06['discount_perc'] < 0.65]


df_065 = df[df['discount_perc'] >= 0.65]
df_065 = df_065[df_065['discount_perc'] < 0.7]


df_07 = df[df['discount_perc'] >= 0.7]
df_07 = df_07[df_07['discount_perc'] < 0.75]

df_075 = df[df['discount_perc'] >= 0.75]
df_075 = df_075[df_075['discount_perc'] < 0.8]


df_08 = df[df['discount_perc'] >= 0.8]
df_08 = df_08[df_08['discount_perc'] < 0.85]

print(df_015.head())
actual_list = [df_zero['actual'].mean(), df_005['actual'].mean(), df_01['actual'].mean(), df_015['actual'].mean(),
               df_02['actual'].mean(), df_025['actual'].mean(), df_03['actual'].mean(),
               df_035['actual'].mean(), df_04['actual'].mean(), df_045['actual'].mean(), 
               df_05['actual'].mean(), df_055['actual'].mean(), df_06['actual'].mean(), 
               df_065['actual'].mean(), df_07['actual'].mean(), df_075['actual'].mean(),
               df_08['actual'].mean()]

predict_list = [df_zero['predict'].mean(), df_005['predict'].mean(), df_01['predict'].mean(), df_015['predict'].mean(),
               df_02['predict'].mean(), df_025['predict'].mean(), df_03['predict'].mean(),
               df_035['predict'].mean(), df_04['predict'].mean(), df_045['predict'].mean(), 
               df_05['predict'].mean(), df_055['predict'].mean(), df_06['predict'].mean(), 
               df_065['predict'].mean(), df_07['predict'].mean(), df_075['predict'].mean(),
               df_08['predict'].mean()]
discount_list = [ 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
print(len(discount_list))
print(len(actual_list))

df_demand = pd.DataFrame({"discount": discount_list,
                            "actual_avg_sppd_demand": actual_list, 'predict': predict_list})
print(df_demand.head())
#df_demand = df_demand[df_demand["discount"]<= 0.4]

plt.scatter(df_demand["discount"], df_demand["actual_avg_sppd_demand"])
plt.scatter(df_demand["discount"], df_demand["predict"])

plt.xlabel("Discount")
plt.ylabel("AVG_SPPD")

#df_demand.to_csv("demand_curve_original_alpha.csv")

print(len(set(Upc)))