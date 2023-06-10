import os
import pandas as pd

def preprocess(file, data):
    with open (file, 'r') as f:
        for line in f:
            # Process each line as needed
            line = line.strip()
            if line:
                components = line.split('\t')
                if len(components) == 3:
                    sentiment = components[1].strip()
                    tweet = components[2].strip()
                    data.append([tweet, sentiment])
    return data

# Get the current directory
current_dir = os.getcwd()

# Define the path to the data directory
data_dir = os.path.join(current_dir, 'data')

# Semeval 2017-4
columns=['text', 'sentiment']
data = []
filenames = ['twitter-2013dev-A.txt', 
             'twitter-2013test-A.txt',
             'twitter-2013train-A.txt',
             'twitter-2014sarcasm-A.txt',
             'twitter-2014test-A.txt',
             'twitter-2015test-A.txt',
             'twitter-2015train-A.txt',
             'twitter-2016devtest-A.txt',
             'twitter-2016test-A.txt',
             'twitter-2016train-A.txt',
             'twitter-2016dev-A.txt']

for filename in filenames:
    file = os.path.join(data_dir, filename)
    data = preprocess(file, data)

df1 = pd.DataFrame(data, columns=columns)

# Sentiment 140

# Read the CSV file
DATASET_ENCODING = 'ISO-8859-1'
DATASET_COLUMNS = ['target','ids','date','flag','user','text']
df2 = pd.read_csv('data/sentiment140.csv', header=None, names=DATASET_COLUMNS, encoding=DATASET_ENCODING)

# Map target values to sentiment labels
sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
df2['sentiment'] = df2['target'].map(sentiment_map)

# Select only 'text' and 'target' columns
df2 = df2[['text', 'sentiment']]

# Concatenate the two DataFrames
df_combined = pd.concat([df1, df2])

# Save the combined DataFrame to a new CSV file
df_combined.to_csv('data/combined.csv', index=False)