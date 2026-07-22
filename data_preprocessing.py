import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

directory = "datagame-2023"
files = os.listdir(directory)

label_encoder = LabelEncoder()

data = {}
#數值化
for file in files:
    if file.endswith(".parquet"):
        parquet_file_path = directory + '/' + file
        df = pd.read_parquet(parquet_file_path)
        
        for column in df:
            if not isinstance(df[column][0], (int, float)):
                df[column] = label_encoder.fit_transform(df[column])
        
        data[file]=df

#合併
# data['label_train_source.parquet']
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song_composer.parquet'], on='song_id', how='inner')
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song_genre.parquet'], on='song_id', how='inner')
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song_lyricist.parquet'], on='song_id', how='inner')
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song_producer.parquet'], on='song_id', how='inner')
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song_titletext.parquet'], on='song_id', how='inner')
# data['label_train_source.parquet'] = pd.merge(data['label_train_source.parquet'], data['meta_song.parquet'], on='song_id', how='inner')

#清洗
for file in data:
    #空值
    df_cleaned = data[file].dropna()
    df_filled = df_cleaned.fillna(df_cleaned.mean())

    #異常值
    # df_cleaned = df[(df['column'] > lower_threshold) & (df['column'] < upper_threshold)]
    # df['column'] = np.where((df['column'] < lower_threshold) | (df['column'] > upper_threshold),
    #                         df['column'].median(), df['column'])

    #重複
    # 删除重复的行
    df_cleaned = df_filled.drop_duplicates()

    # 使用特定的唯一标识符去除重复
    # df_cleaned = df_filled.drop_duplicates(subset=['column'])

    #統一
    # 统一字符串的大小写
    # df['column'] = df['column'].str.lower()

    # 使用映射表进行标准化
    # mapping = {'male': 'M', 'female': 'F'}
    # df['gender'] = df['gender'].map(mapping)

    #刪除不符
    # 根据业务规则删除或修复数据
    # df_cleaned = df[(df['value'] >= lower_limit) & (df['value'] <= upper_limit)]

for f in data:
    data[f].to_parquet(f, index=False)