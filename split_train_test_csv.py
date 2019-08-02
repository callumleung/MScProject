import pandas as pd
from sklearn.model_selection import train_test_split


mixed_csv = pd.read_csv("20_examples.csv")

# shuffle the list of images
mixed_csv = mixed_csv.sample(frac=1).reset_index(drop=True)


train_df, test_df = train_test_split(mixed_csv, test_size=0.25)
# index of row 75% through the csv
# this yields 75% train and 25% test
# end_train_index = int(len(mixed_csv) * 0.75)
# train_df = mixed_csv.iloc[:, :end_train_index]
# test_df = mixed_csv.iloc[:, end_train_index:]


train_df.to_csv("20_examples_train.csv", index=None)
test_df.to_csv("20_examples_test.csv", index=None)

