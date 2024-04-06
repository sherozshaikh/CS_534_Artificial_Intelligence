import pandas as pd

cards = pd.read_csv('sd254_cards.csv')
users = pd.read_csv('sd254_users.csv')
transactions = pd.read_csv('credit_card_transactions-ibm_v2.csv', chunksize=100000)

users_and_cards = pd.merge(users, cards, right_on='User', left_index=True)

dataset = pd.merge(transactions.read(), users_and_cards, left_on=['User', 'Card'], right_on=['User', 'CARD INDEX'])

#print(dataset.head(10).to_string())

dataset.to_csv('total_dataset.csv')



