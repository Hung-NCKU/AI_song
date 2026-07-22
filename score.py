import pandas as pd
train = pd.read_parquet('datagame-2023/label_train_source.parquet')
train1 = pd.read_parquet('datagame-2023/label_train_target.parquet')
test = pd.read_parquet('datagame-2023/label_test_source.parquet')
train = pd.concat([train, train1], ignore_index=True)
train = pd.concat([train, test], ignore_index=True)

column_to_count = train['song_id'] 
value_counts = column_to_count.value_counts()
percentage = value_counts / len(train)
train['score'] = train['song_id'].map(percentage)

train.to_parquet('score.parquet', index=False)
print(train)