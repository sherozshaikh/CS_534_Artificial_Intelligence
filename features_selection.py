import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

ss = StandardScaler()
mm = MinMaxScaler()

transactions = pd.read_csv('cleaned_transactions (1).csv')

X_lat_long = transactions[['Latitude','Longitude']].values
kmeans = KMeans(n_clusters=4,random_state=123)
kmeans.fit(X_lat_long)
transactions['Cluster_ID'] = kmeans.labels_

le_encoded = {'Card Brand': {'Amex': 0, 'Discover': 1, 'Mastercard': 2, 'Visa': 3}, 'Card Type': {'Credit': 0, 'Debit': 1, 'Debit (Prepaid)': 2}, 'Has Chip': {'NO': 0, 'YES': 1}, 'Gender': {'Female': 0, 'Male': 1}, 'State': {'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'CA': 4, 'CO': 5, 'CT': 6, 'DE': 7, 'FL': 8, 'GA': 9, 'HI': 10, 'IA': 11, 'ID': 12, 'IL': 13, 'IN': 14, 'KS': 15, 'KY': 16, 'LA': 17, 'MA': 18, 'MD': 19, 'ME': 20, 'MI': 21, 'MN': 22, 'MO': 23, 'MS': 24, 'MT': 25, 'NC': 26, 'ND': 27, 'NE': 28, 'NH': 29, 'NJ': 30, 'NM': 31, 'NV': 32, 'NY': 33, 'OH': 34, 'OK': 35, 'OR': 36, 'PA': 37, 'RI': 38, 'SC': 39, 'SD': 40, 'TN': 41, 'TX': 42, 'UT': 43, 'VA': 44, 'VT': 45, 'WA': 46, 'WI': 47, 'WV': 48, 'WY': 49}, 'Use Chip': {'Chip Transaction': 0, 'Online Transaction': 1, 'Swipe Transaction': 2}, 'Merchant State': {'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'Algeria': 4, 'CA': 5, 'CO': 6, 'CT': 7, 'Canada': 8, 'China': 9, 'Colombia': 10, 'Costa Rica': 11, 'Czech Republic': 12, 'DC': 13, 'DE': 14, 'Dominican Republic': 15, 'Estonia': 16, 'FL': 17, 'Fiji': 18, 'France': 19, 'GA': 20, 'Germany': 21, 'Greece': 22, 'Guatemala': 23, 'HI': 24, 'Haiti': 25, 'Hong Kong': 26, 'IA': 27, 'ID': 28, 'IL': 29, 'IN': 30, 'India': 31, 'Italy': 32, 'Jamaica': 33, 'Japan': 34, 'KS': 35, 'KY': 36, 'LA': 37, 'Luxembourg': 38, 'MA': 39, 'MD': 40, 'ME': 41, 'MI': 42, 'MN': 43, 'MO': 44, 'MS': 45, 'MT': 46, 'Malaysia': 47, 'Mexico': 48, 'NC': 49, 'ND': 50, 'NE': 51, 'NH': 52, 'NJ': 53, 'NM': 54, 'NV': 55, 'NY': 56, 'Netherlands': 57, 'Nigeria': 58, 'Norway': 59, 'OH': 60, 'OK': 61, 'OR': 62, 'PA': 63, 'Pakistan': 64, 'Peru': 65, 'Philippines': 66, 'Poland': 67, 'Portugal': 68, 'RI': 69, 'SC': 70, 'SD': 71, 'Singapore': 72, 'South Africa': 73, 'South Korea': 74, 'Spain': 75, 'Switzerland': 76, 'TN': 77, 'TX': 78, 'Taiwan': 79, 'Thailand': 80, 'The Bahamas': 81, 'Turkey': 82, 'Tuvalu': 83, 'UT': 84, 'United Kingdom': 85, 'VA': 86, 'VT': 87, 'Vatican City': 88, 'WA': 89, 'WI': 90, 'WV': 91, 'WY': 92, 'nan': 93}, 'Errors?': {'Bad CVV': 0, 'Bad CVV,Insufficient Balance': 1, 'Bad CVV,Technical Glitch': 2, 'Bad Card Number': 3, 'Bad Card Number,Insufficient Balance': 4, 'Bad Expiration': 5, 'Bad Expiration,Technical Glitch': 6, 'Bad PIN': 7, 'Bad PIN,Insufficient Balance': 8, 'Bad Zipcode': 9, 'Insufficient Balance': 10, 'Technical Glitch': 11, 'nan': 12}, 'State_US_region': {'Midwest': 0, 'Northeast': 1, 'Southeast': 2, 'Southwest': 3, 'West': 4}, 'Merchant Country': {'Algeria': 0, 'Canada': 1, 'China': 2, 'Colombia': 3, 'Costa Rica': 4, 'Czech Republic': 5, 'Dominican Republic': 6, 'Estonia': 7, 'Fiji': 8, 'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13, 'Hong Kong': 14, 'India': 15, 'Italy': 16, 'Jamaica': 17, 'Japan': 18, 'Luxembourg': 19, 'Malaysia': 20, 'Mexico': 21, 'Netherlands': 22, 'Nigeria': 23, 'Norway': 24, 'Online': 25, 'Pakistan': 26, 'Peru': 27, 'Philippines': 28, 'Poland': 29, 'Portugal': 30, 'Singapore': 31, 'South Africa': 32, 'South Korea': 33, 'Spain': 34, 'Switzerland': 35, 'Taiwan': 36, 'Thailand': 37, 'The Bahamas': 38, 'Turkey': 39, 'Tuvalu': 40, 'USA': 41, 'United Kingdom': 42, 'Vatican City': 43}, 'MCC_CAT': {'Amusement and entertainment': 0, 'Automobiles and vehicles': 1, 'Business services': 2, 'Clothing outlets': 3, 'Contracted services': 4, 'Government services': 5, 'Miscellaneous outlets': 6, 'Professional services and membership organizations': 7, 'Repair services': 8, 'Reserved for private use': 9, 'Retail outlets': 10, 'Service providers': 11, 'Transportation': 12, 'Utilities': 13}, 'Card Brand Type': {'Amex Credit': 0, 'Discover Credit': 1, 'Mastercard Credit': 2, 'Mastercard Debit': 3, 'Mastercard Debit (Prepaid)': 4, 'Visa Credit': 5, 'Visa Debit': 6, 'Visa Debit (Prepaid)': 7}}
for column,encoding in le_encoded.items():
    transactions['lb_enc_'+column] = transactions[column].map(encoding)

ss = ss.fit_transform(transactions[['Cards Issued','Per Capita Income - Zipcode','Yearly Income - Person','Num Credit Cards','Amount','Year','#Pin Changed by User','Avg_MCC','Avg_State','Day_of_Week','Month_sin','Month_cos','Day_of_Week_sin','Day_of_Week_cos','User_Active_Past_3_Months','User_Active_Past_6_Months','User_Active_Past_9_Months','User_Active_Past_12_Months','User_Active_Past_15_Months','User_Active_Past_18_Months','User_Active_Past_21_Months','User_Active_Past_24_Months','FICO_Score_Binned_Ordinal','User_Age_Binned_Ordinal','Debt_Credit_Limit_Ratio','Yearly_Income_To_Total_Debt_Ratio','Yearly_Income_To_Credit_Limit_Ratio','Num_Of_Cards_Relative_To_Median']])
mm = mm.fit_transform(transactions[['Credit Limit','Year PIN last Changed','Current Age','Retirement Age','Birth Year','Latitude','Longitude','Total Debt','FICO Score','Month','Day','MCC','#Transaction by User','Hour','Day_sin','Day_cos','Hour_sin','Hour_cos','User_Amount_Spent_Past_3_Months','User_Amount_Spent_Past_6_Months','User_Amount_Spent_Past_9_Months','User_Amount_Spent_Past_12_Months','User_Amount_Spent_Past_15_Months','User_Amount_Spent_Past_18_Months','User_Amount_Spent_Past_21_Months','User_Amount_Spent_Past_24_Months','Expiration Year','Account_Open_Year','PIN last Changed','Account Open Age','Years in Card Expiration','Retirement to Current','Credit_Limit_Utilization','Years PIN Change After Account Open','Months_Since_Account_Open']])
ss.columns = ['ss_enc_'+i for i in ss.columns]
mm.columns = ['mm_enc_'+i for i in mm.columns]
transactions = pd.concat([transactions,ss,mm],axis=1)

X_train,y_train = transactions,transactions[['Is Fraud?']].values
X_train = X_train.drop(columns=['Is Fraud?'])
y_train = y_train.reshape(-1)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=123,stratify=y)

revised_mapping_search = {

    "#Pin Changed by User": [["ss_enc_#Pin Changed by User"]],
    "#Transaction by User": [["mm_enc_#Transaction by User"]],
    "Account Open Age": [["mm_enc_Account Open Age"]],
    "Account_Open_Year": [["mm_enc_Account_Open_Year"]],
    "Amount": [["ss_enc_Amount"]],
    "Avg_MCC": [["ss_enc_Avg_MCC"]],
    "Avg_State": [["ss_enc_Avg_State"]],
    "Birth Year": [["mm_enc_Birth Year"]],
    "Card Brand Type": [["lb_enc_Card Brand Type"]],
    "Card Brand": [["lb_enc_Card Brand"]],
    "Card Number": [["Card Number_F4"]],
    "Card Type": [["lb_enc_Card Type"]],
    "Cards Issued": [["Cards Issued"]],
    "Cluster_ID": [["Cluster_ID"]],
    "Credit Limit": [["mm_enc_Credit Limit"]],
    "Credit_Limit_Utilization": [["mm_enc_Credit_Limit_Utilization"]],
    "Current Age": [["mm_enc_Current Age"],["User_Age_Binned_Ordinal"],["ss_enc_User_Age_Binned_Ordinal"]],
    "Day": [["Day"],["mm_enc_Day"],["mm_enc_Day_sin"],["mm_enc_Day_cos"]],
    "Day_of_Week": [["Day_of_Week"],["ss_enc_Day_of_Week"]],
    "Debt_Credit_Limit_Ratio": [["ss_enc_Debt_Credit_Limit_Ratio"]],
    "Errors?": [["Errors_Labels"],["Error_Ratio"]],
    "Expiration Year": [["mm_enc_Expiration Year"]],
    "FICO Score": [["FICO Score"],["FICO_Score_Binned_Ordinal"],["mm_enc_FICO Score"]],
    "Gender": [["lb_enc_Gender"]],
    "Has Chip": [["Has Chip_Binary"]],
    "Hour": [["Hour"],["mm_enc_Hour"],["mm_enc_Hour_sin"],["mm_enc_Hour_cos"]],
    "Latitude": [["mm_enc_Latitude"]],
    "Longitude": [["mm_enc_Longitude"]],
    "MCC": [["MCC"],["mm_enc_MCC"],["lb_enc_MCC_CAT"]],
    "Merchant Country": [["lb_enc_Merchant Country"]],
    "Merchant State": [["lb_enc_Merchant State"]],
    "Month": [["Month"],["ss_enc_Month_sin"],["ss_enc_Month_cos"],],
    "Months_Since_Account_Open": [["mm_enc_Months_Since_Account_Open"]],
    "Num Credit Cards": [["Num Credit Cards"]],
    "Num_Of_Cards_Relative_To_Median": [["ss_enc_Num_Of_Cards_Relative_To_Median"]],
    "Per Capita Income - Zipcode": [["ss_enc_Per Capita Income - Zipcode"]],
    "PIN last Changed": [["mm_enc_PIN last Changed"]],
    "Retirement Age": [["mm_enc_Retirement Age"]],
    "Retirement to Current": [["mm_enc_Retirement to Current"]],
    "State": [["lb_enc_State",]],
    "State_US_region": [["lb_enc_State_US_region"]],
    "Total Debt": [["mm_enc_Total Debt"]],
    "Use Chip": [["lb_enc_Use Chip"]],
    "User_Active_Past_12_Months": [["ss_enc_User_Active_Past_12_Months"]],
    "User_Active_Past_15_Months": [["ss_enc_User_Active_Past_15_Months"]],
    "User_Active_Past_18_Months": [["ss_enc_User_Active_Past_18_Months"]],
    "User_Active_Past_21_Months": [["ss_enc_User_Active_Past_21_Months"]],
    "User_Active_Past_24_Months": [["ss_enc_User_Active_Past_24_Months"]],
    "User_Active_Past_3_Months": [["ss_enc_User_Active_Past_3_Months"]],
    "User_Active_Past_6_Months": [["ss_enc_User_Active_Past_6_Months"]],
    "User_Active_Past_9_Months": [["ss_enc_User_Active_Past_9_Months"]],
    "User_Amount_Spent_Past_12_Months": [["mm_enc_User_Amount_Spent_Past_12_Months"]],
    "User_Amount_Spent_Past_15_Months": [["mm_enc_User_Amount_Spent_Past_15_Months"]],
    "User_Amount_Spent_Past_18_Months": [["mm_enc_User_Amount_Spent_Past_18_Months"]],
    "User_Amount_Spent_Past_21_Months": [["mm_enc_User_Amount_Spent_Past_21_Months"]],
    "User_Amount_Spent_Past_24_Months": [["mm_enc_User_Amount_Spent_Past_24_Months"]],
    "User_Amount_Spent_Past_3_Months": [["mm_enc_User_Amount_Spent_Past_3_Months"]],
    "User_Amount_Spent_Past_6_Months": [["mm_enc_User_Amount_Spent_Past_6_Months"]],
    "User_Amount_Spent_Past_9_Months": [["mm_enc_User_Amount_Spent_Past_9_Months"]],
    "Year PIN last Changed": [["mm_enc_Year PIN last Changed"]],
    "Year": [["ss_enc_Year",]],
    "Yearly Income - Person": [["ss_enc_Yearly Income - Person"]],
    "Yearly_Income_To_Credit_Limit_Ratio": [["ss_enc_Yearly_Income_To_Credit_Limit_Ratio"]],
    "Yearly_Income_To_Total_Debt_Ratio": [["ss_enc_Yearly_Income_To_Total_Debt_Ratio"]],
    "Years in Card Expiration": [["mm_enc_Years in Card Expiration"]],
    "Years PIN Change After Account Open": [["mm_enc_Years PIN Change After Account Open"]],

}

all_combinations = []
for val_1 in revised_mapping_search["#Pin Changed by User"]:
    for val_2 in revised_mapping_search["#Transaction by User"]:
        for val_3 in revised_mapping_search["Account Open Age"]:
            for val_4 in revised_mapping_search["Account_Open_Year"]:
                for val_5 in revised_mapping_search["Amount"]:
                    for val_6 in revised_mapping_search["Avg_MCC"]:
                        for val_7 in revised_mapping_search["Avg_State"]:
                            for val_8 in revised_mapping_search["Birth Year"]:
                                for val_9 in revised_mapping_search["Card Brand Type"]:
                                    for val_10 in revised_mapping_search["Card Brand"]:
                                        for val_11 in revised_mapping_search["Card Number"]:
                                            for val_12 in revised_mapping_search["Card Type"]:
                                                for val_13 in revised_mapping_search["Cards Issued"]:
                                                    for val_14 in revised_mapping_search["Cluster_ID"]:
                                                        for val_15 in revised_mapping_search["Credit Limit"]:
                                                            for val_16 in revised_mapping_search["Credit_Limit_Utilization"]:
                                                                for val_17 in revised_mapping_search["Current Age"]:
                                                                    for val_18 in revised_mapping_search["Day"]:
                                                                        for val_19 in revised_mapping_search["Day_of_Week"]:
                                                                            for val_20 in revised_mapping_search["Debt_Credit_Limit_Ratio"]:
                                                                                for val_21 in revised_mapping_search["Errors?"]:
                                                                                    for val_22 in revised_mapping_search["Expiration Year"]:
                                                                                        for val_23 in revised_mapping_search["FICO Score"]:
                                                                                            for val_24 in revised_mapping_search["Gender"]:
                                                                                                for val_25 in revised_mapping_search["Has Chip"]:
                                                                                                    for val_26 in revised_mapping_search["Hour"]:
                                                                                                        for val_27 in revised_mapping_search["Latitude"]:
                                                                                                            for val_28 in revised_mapping_search["Longitude"]:
                                                                                                                for val_29 in revised_mapping_search["MCC"]:
                                                                                                                    for val_30 in revised_mapping_search["Merchant Country"]:
                                                                                                                        for val_31 in revised_mapping_search["Merchant State"]:
                                                                                                                            for val_32 in revised_mapping_search["Month"]:
                                                                                                                                for val_33 in revised_mapping_search["Months_Since_Account_Open"]:
                                                                                                                                    for val_34 in revised_mapping_search["Num Credit Cards"]:
                                                                                                                                        for val_35 in revised_mapping_search["Num_Of_Cards_Relative_To_Median"]:
                                                                                                                                            for val_36 in revised_mapping_search["Per Capita Income - Zipcode"]:
                                                                                                                                                for val_37 in revised_mapping_search["PIN last Changed"]:
                                                                                                                                                    for val_38 in revised_mapping_search["Retirement Age"]:
                                                                                                                                                        for val_39 in revised_mapping_search["Retirement to Current"]:
                                                                                                                                                            for val_40 in revised_mapping_search["State"]:
                                                                                                                                                                for val_41 in revised_mapping_search["State_US_region"]:
                                                                                                                                                                    for val_42 in revised_mapping_search["Total Debt"]:
                                                                                                                                                                        for val_43 in revised_mapping_search["Use Chip"]:
                                                                                                                                                                            for val_44 in revised_mapping_search["User_Active_Past_12_Months"]:
                                                                                                                                                                                for val_45 in revised_mapping_search["User_Active_Past_15_Months"]:
                                                                                                                                                                                    for val_46 in revised_mapping_search["User_Active_Past_18_Months"]:
                                                                                                                                                                                        for val_47 in revised_mapping_search["User_Active_Past_21_Months"]:
                                                                                                                                                                                            for val_48 in revised_mapping_search["User_Active_Past_24_Months"]:
                                                                                                                                                                                                for val_49 in revised_mapping_search["User_Active_Past_3_Months"]:
                                                                                                                                                                                                    for val_50 in revised_mapping_search["User_Active_Past_6_Months"]:
                                                                                                                                                                                                        for val_51 in revised_mapping_search["User_Active_Past_9_Months"]:
                                                                                                                                                                                                            for val_52 in revised_mapping_search["User_Amount_Spent_Past_12_Months"]:
                                                                                                                                                                                                                for val_53 in revised_mapping_search["User_Amount_Spent_Past_15_Months"]:
                                                                                                                                                                                                                    for val_54 in revised_mapping_search["User_Amount_Spent_Past_18_Months"]:
                                                                                                                                                                                                                        for val_55 in revised_mapping_search["User_Amount_Spent_Past_21_Months"]:
                                                                                                                                                                                                                            for val_56 in revised_mapping_search["User_Amount_Spent_Past_24_Months"]:
                                                                                                                                                                                                                                for val_57 in revised_mapping_search["User_Amount_Spent_Past_3_Months"]:
                                                                                                                                                                                                                                    for val_58 in revised_mapping_search["User_Amount_Spent_Past_6_Months"]:
                                                                                                                                                                                                                                        for val_59 in revised_mapping_search["User_Amount_Spent_Past_9_Months"]:
                                                                                                                                                                                                                                            for val_60 in revised_mapping_search["Year PIN last Changed"]:
                                                                                                                                                                                                                                                for val_61 in revised_mapping_search["Year"]:
                                                                                                                                                                                                                                                    for val_62 in revised_mapping_search["Yearly Income - Person"]:
                                                                                                                                                                                                                                                        for val_63 in revised_mapping_search["Yearly_Income_To_Credit_Limit_Ratio"]:
                                                                                                                                                                                                                                                            for val_64 in revised_mapping_search["Yearly_Income_To_Total_Debt_Ratio"]:
                                                                                                                                                                                                                                                                for val_65 in revised_mapping_search["Years in Card Expiration"]:
                                                                                                                                                                                                                                                                    for val_66 in revised_mapping_search["Years PIN Change After Account Open"]:
                                                                                                                                                                                                                                                                        combination = val_1 + val_2 + val_3 + val_4 + val_5 + val_6 + val_7 + val_8 + val_9 + val_10 + val_11 + val_12 + val_13 + val_14 + val_15 + val_16 + val_17 + val_18 + val_19 + val_20 + val_21 + val_22 + val_23 + val_24 + val_25 + val_26 + val_27 + val_28 + val_29 + val_30 + val_31 + val_32 + val_33 + val_34 + val_35 + val_36 + val_37 + val_38 + val_39 + val_40 + val_41 + val_42 + val_43 + val_44 + val_45 + val_46 + val_47 + val_48 + val_49 + val_50 + val_51 + val_52 + val_53 + val_54 + val_55 + val_56 + val_57 + val_58 + val_59 + val_60 + val_61 + val_62 + val_63 + val_64 + val_65 + val_66
                                                                                                                                                                                                                                                                        all_combinations.append(combination)

for i in all_combinations:
    X_train_temp = X_train[i]
    X_test_temp = X_test[i]
    rf_classifier = RandomForestClassifier(random_state=123)
    rf_classifier.fit(X_train_temp,y_train)
    y_pred_rf = rf_classifier.predict(X_test_temp)
    accuracy = accuracy_score(y_test,y_pred_rf)
    print(i,accuracy,rf_classifier.feature_names_in_,rf_classifier.feature_importances_)

X_train = X_train[['mm_enc_#Transaction by User','lb_enc_Merchant Country','mm_enc_User_Amount_Spent_Past_15_Months','mm_enc_User_Amount_Spent_Past_18_Months','mm_enc_User_Amount_Spent_Past_9_Months','mm_enc_User_Amount_Spent_Past_24_Months','mm_enc_User_Amount_Spent_Past_6_Months','mm_enc_User_Amount_Spent_Past_12_Months','mm_enc_User_Amount_Spent_Past_21_Months','ss_enc_Avg_MCC','mm_enc_User_Amount_Spent_Past_3_Months','lb_enc_Merchant State','ss_enc_Avg_State']]
X_test = X_test[['mm_enc_#Transaction by User','lb_enc_Merchant Country','mm_enc_User_Amount_Spent_Past_15_Months','mm_enc_User_Amount_Spent_Past_18_Months','mm_enc_User_Amount_Spent_Past_9_Months','mm_enc_User_Amount_Spent_Past_24_Months','mm_enc_User_Amount_Spent_Past_6_Months','mm_enc_User_Amount_Spent_Past_12_Months','mm_enc_User_Amount_Spent_Past_21_Months','ss_enc_Avg_MCC','mm_enc_User_Amount_Spent_Past_3_Months','lb_enc_Merchant State','ss_enc_Avg_State']]

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test,y_pred_nb)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

lr_classifier = LogisticRegression(random_state=123,solver='lbfgs',C=0.1,max_iter=1000)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_lr = accuracy_score(y_test,y_pred_lr)

rf_classifier = RandomForestClassifier(random_state=123)
rf_classifier.fit(X_train,y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test,y_pred_rf)

