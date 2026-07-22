import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np


data = pd.read_parquet('train.parquet')
user_features = pd.read_parquet('test.parquet')

label_encoder_song_main = LabelEncoder()
label_encoder_song = LabelEncoder()
# for i in data:
#     print(i)

data['song_id'] = label_encoder_song_main.fit_transform(data['song_id'])
label_encoder_geid = LabelEncoder()
data['genre_id'] = label_encoder_geid.fit_transform(data['genre_id'])
label_encoder_coid = LabelEncoder()
data['composer_id'] = label_encoder_coid.fit_transform(data['composer_id'])
label_encoder_lyid = LabelEncoder()
data['lyricist_id'] = label_encoder_lyid.fit_transform(data['lyricist_id'])
label_encoder_prid = LabelEncoder()
data['producer_id'] = label_encoder_prid.fit_transform(data['producer_id'])
label_encoder_txid = LabelEncoder()
data['title_text_id'] = label_encoder_txid.fit_transform(data['title_text_id'])
label_encoder_t = LabelEncoder()
data['album_month'] = label_encoder_t.fit_transform(data['album_month'])
# 进行推荐，生成用户对歌曲的预测评分

user_features['song_id'] = label_encoder_song.fit_transform(user_features['song_id'])
user_features['genre_id'] = label_encoder_geid.fit_transform(user_features['genre_id'])
user_features['composer_id'] = label_encoder_coid.fit_transform(user_features['composer_id'])
user_features['lyricist_id'] = label_encoder_lyid.fit_transform(user_features['lyricist_id'])
user_features['producer_id'] = label_encoder_prid.fit_transform(user_features['producer_id'])
user_features['title_text_id'] = label_encoder_txid.fit_transform(user_features['title_text_id'])
user_features['album_month'] = label_encoder_t.fit_transform(user_features['album_month'])



features = ['session_id', 'song_id', 'unix_played_at', 'play_status', 'login_type', 'listening_order', 'artist_id', 'song_length', 'album_id', 'language_id', 'album_month', 'composer_id', 'genre_id', 'lyricist_id', 'producer_id', 'title_text_id']
target = 'score'  # 假设这是用户对歌曲的评分

# 划分数据集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = XGBRegressor()
model.fit(train_data[features], train_data[target])

# 在测试集上进行预测
predictions = model.predict(test_data[features])

# 评估模型性能
mse = mean_squared_error(test_data[target], predictions)
print(f'Mean Squared Error: {mse}')


song_predictions = model.predict(user_features)
# song_predictions = label_encoder.inverse_transform(song_predictions)

new_attribute_name = 'score'
new_attribute_values = song_predictions  # 根据你的需求修改
user_features[new_attribute_name] = new_attribute_values
user_features.to_parquet('final.parquet', index=False)

sorted_df = user_features.sort_values(by=['session_id', 'score'], ascending=[True, False])
top_five_per_session = sorted_df.groupby('session_id').head(5)

top_five_song_ids = top_five_per_session['song_id'].tolist()
top_five_session_ids = top_five_per_session['session_id'].tolist()

columns = ['session_id', 'top1', 'top2', 'top3', 'top4', 'top5']
df = pd.DataFrame(columns=columns)

for i in range(0,len(top_five_session_ids),5):
    # new = {'session_id':int(top_five_song_ids[i]), 'top1':label_encoder.inverse_transform(top_five_session_ids[i]), 'top2':label_encoder.inverse_transform(top_five_session_ids[i+1]), 'top3':label_encoder.inverse_transform(top_five_session_ids[i+2]), 'top4':label_encoder.inverse_transform(top_five_session_ids[i+3]), 'top5':label_encoder.inverse_transform(top_five_session_ids[i+4])}
    new = {'session_id':top_five_song_ids[i], 'top1':top_five_session_ids[i], 'top2':top_five_session_ids[i+1], 'top3':top_five_session_ids[i+2], 'top4':top_five_session_ids[i+3], 'top5':top_five_session_ids[i+4]}
    new_row = pd.DataFrame([new])
    df = pd.concat([df, new_row], ignore_index=True)

df['top1'] = label_encoder_song.inverse_transform(df['top1'])
df['top2'] = label_encoder_song.inverse_transform(df['top2'])
df['top3'] = label_encoder_song.inverse_transform(df['top3'])
df['top4'] = label_encoder_song.inverse_transform(df['top4'])
df['top5'] = label_encoder_song.inverse_transform(df['top5'])

df.to_parquet('submission_f.parquet', index=False)


# 根据预测评分生成推荐
print(f'Recommended Songs: {song_predictions}')
