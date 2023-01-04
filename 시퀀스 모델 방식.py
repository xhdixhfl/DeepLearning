!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz

# 필요없는 저장소 삭제
! rm -r aclImdb/train/unsup
# 파일 내용 살펴보기
!cat aclImdb/train/pos/4077_10.txt

# 검증세트 만들기
# 훈련파일의 20%, 새 저장소에
import os, pathlib, shutil, random

base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for cate in ("neg", "pos"):
  os.makedirs(val_dir / cate)
  files = os.listdir(train_dir / cate)
  random.Random(1337).shuffle(files)
  n_val_samples = int(0.2 * len(files))  # 훈련파일 중 20%
  val_files = files[-n_val_samples:]
  for fname in val_files:  # 파일 위치
    shutil.move(train_dir / cate / fname,
                val_dir /  cate / fname)
    
# 훈련, 검증, 테스트 셋을 위한 데이터셋 객체 
from tensorflow import keras

b_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size = b_size
)
val_ds =  keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size = b_size
)
test_ds =  keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size = b_size
)

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
t_only_train_ds = train_ds.map(lambda x, y: x)
from tensorflow.keras import layers
# 정수 시퀀스 데이터셋 
max_length = 600
max_tokens = 20000
text_vec = layers.TextVectorization(
    max_tokens = max_tokens,
    output_mode = "int",
    output_sequence_length = max_length,
)
text_vec.adapt(t_only_train_ds)
int_train_ds = train_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4,
)
int_val_ds = val_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4,
)
int_test_ds = test_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4,
)

# 원-핫 인코딩된 벡트 시퀀스 모델
import tensorflow as tf

inputs = keras.Input(shape=(None,), dtype='int64')
embedded = tf.one_hot(inputs, depth = max_tokens) # 인코딩
x = layers.Bidirectional(layers.LSTM(32))(embedded)# 양방향 LSTM층
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x) # 분류 층 추가
model = keras.Model(inputs, outputs)
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.summary()

# 모델 훈련
cb = [
    keras.callbacks.ModelCheckpoint('one_hot_bidir_lstm.keras',
                                    save_best_only = True)
]
model.fit(int_train_ds, 
          validation_data = int_val_ds,
          epochs = 10,
          callbacks = cb,)
model = keras.models.load_model('one_hot_bidir_lstm.keras')
print(f'테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}')

# embedding층 만들기
embedding_layer = layers.Embedding(input_dim = max_tokens, 
                                   output_dim = 256)
# 임배딩 층은 적어도 2개의 매개변수가 필요함..

# 모델링
inputs = keras.Input(shape=(None,), dtype='int64')
embedded = layers.Embedding(input_dim = max_tokens, 
                            output_dim = 256)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)# 양방향 LSTM층
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x) # 분류 층 추가
model = keras.Model(inputs, outputs)
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.summary()

callback = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_lstm.keras',
                                    save_best_only = True)
]
model.fit(int_train_ds,
          validation_data = int_val_ds,
          epochs = 10, 
          callbacks = callback)
model = keras.models.load_model("embeddings_bidir_lstm.keras")
print(f'테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}')

# masking을 활성화한 모델링
inputs = keras.Input(shape=(None,), dtype='int64')
embedded = layers.Embedding(input_dim = max_tokens, 
                            output_dim = 256,
                            mask_zero=True)(inputs) # 마스킹 활성화
x = layers.Bidirectional(layers.LSTM(32))(embedded)# 양방향 LSTM층
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x) # 분류 층 추가
model = keras.Model(inputs, outputs)
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'embeddings_bidir_lstm_with_masking.keras',
         save_best_only = True)
]
model.fit(int_train_ds,
          validation_data = int_val_ds,
          epochs = 10, 
          callbacks = callbacks)
model = keras.models.load_model("embeddings_bidir_lstm_with_masking.keras")
print(f'테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}')

# GloVe 임베딩
!wget https://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip

# glove 단어 임베딩 파일 파싱
import numpy as np

path_to_glove_file = 'glove.6B.100d.txt'
embeddings_index = {}
with open(path_to_glove_file) as f:
  for line in f:
    word, coefs = line.split(maxsplit = 1)
    coefs = np.fromstring(coefs, 'f', sep = ' ')
    embeddings_index[word] = coefs

print(f'단어 벡터 수: {len(embeddings_index)}')

# glove단어 임베딩 행렬
embedding_dim = 100

voca = text_vec.get_vocabulary() # TextVocabulary층에서 인덱싱된 단어 추출
word_index = dict(zip(voca, range(len(voca)))) # 단어와 인덱스 매핑

embedding_matrix = np.zeros((max_tokens, embedding_dim)) # 벡터 채울 행렬
for word, i in word_index.items():
  if i < max_tokens:
    embedding_vec = embeddings_index.get(word) 
    if embedding_vec is not None:  # 인덱스 i에 대한 단어 백터를 만들어둔 행렬의 i번쨰에 채움 인덱스 없으면 0
      embedding_matrix[i] = embedding_vec
      
#constant 초기화를 사용한 임베딩 층
embedding_layer = layers.Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer = keras.initializers.Constant(embedding_matrix),
    trainable = False,
    mask_zero = True
)

# 사전훈련된 임베딩(GloVe)사용한 모델링
inputs = keras.Input(shape=(None,), dtype = 'int64')
embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('glove_embeddings_sequence_model.keras',
                                    save_best_only = True)
]
model.fit(int_train_ds, 
          validation_data = int_val_ds,
          epochs = 10,
          callbacks = callbacks)
model = keras.models.load_model('glove_embeddings_sequence_model.keras')
print(f'테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}')


