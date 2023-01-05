# 데이터 셋
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
# 클래스 정의
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim # 입력 토큰 벡터의 크기 
    self.dense_dim = dense_dim # 내부 밀집 층의 크기
    self.num_heads = num_heads # 어텐션 해드 개수
    self.attention = layers.MultiHeadAttention(
        num_heads = num_heads, key_dim = embed_dim
    )
    self.dense_proj = keras.Sequential(
        [layers.Dense(dense_dim, activation = 'relu'),
         layers.Dense(embed_dim),]
    )
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()

# 연산 수행
  def call(self, inputs, mask = None): 
    if mask is not None:   # 임베딩 층에서 생성하는 마스크는 2차원이지만 어텐션 층은 3또는 4차원을 기대해서
      mask = mask[:, tf.newaxis, :]  # 랭크를 늘리는 작업
    attention_output = self.attention(
        inputs, inputs, attention_mask = mask
    )
    proj_input = self.layernorm_1(inputs + attention_output)
    proj_output = self.dense_proj(proj_input)
    return self.layernorm_2(proj_input + proj_output)

# 모델 저장을 위한 직렬화(직렬형태여애 저장이 가능)
  def get_config(self):
    config = super().get_config()
    config.update({
        'embed_dim' : self.embed_dim,
        'num_heads' : self.num_heads,
        'dense_dim' : self.dense_dim,
    })
    return config

  # 설정값
vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

# 트랜스 포머 인코더를 사용하는 텍스트 분류기
inputs = keras.Input(shape=(None,), dtype = 'int64')
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x) # 각 시퀀스를 하나의 벡터로 만드는것을 수행
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

# 모델 훈련 및 평가
cb = [
    keras.callbacks.ModelCheckpoint('transformer_encoder.keras',
                                    save_best_only = True)
]
model.fit(int_train_ds, validation_data = int_val_ds, epochs = 20,
          callbacks = cb)
model = keras.models.load_model(
    'transformer_encoder.keras',
    custom_objects = {'TransformerEncoder':TransformerEncoder}
)
print(f'테스트 정확도 : {model.evaluate(int_test_ds)[1]:.3f}')

# 서브클래싱으로 위치임베딩 구현
class PositionalEmbedding(layers.Layer):
  def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.token_embeddings = layers.Embedding(
        input_dim = input_dim, output_dim = output_dim
    )
    self.position_embeddings = layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.sequence_length = sequence_length
    self.input_dim = input_dim
    self.output_dim = output_dim

  def call(self, inputs):
    length = tf.shape(inputs)[-1]
    positions = tf.range(start = 0, limit = length, delta = 1)
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions = self.position_embeddings(positions)
    return embedded_tokens + embedded_positions 

  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_dim' : self.output_dim,
        'sequence_length' : self.sequence_length,
        'input_dim' : self.input_dim,
    })
    return config
  
# 트랜스포머 인코더와 위치 임베딩 합치기
vocab_size = 20000
sequence_length = 600
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype = 'int64')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs) # 위치 임베딩
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x) # 각 시퀀스를 하나의 벡터로 만드는것을 수행
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callback = [
    keras.callbacks.ModelCheckpoint('full_transformer_encoder.keras',
                                    save_best_only = True)
]
model.fit(int_train_ds, validation_data = int_val_ds, epochs = 20,
          callbacks = callback)
model = keras.models.load_model(
    'full_transformer_encoder.keras',
    custom_objects = {'TransformerEncoder':TransformerEncoder,
                      'PositionalEmbedding': PositionalEmbedding}
)
print(f'테스트 정확도 : {model.evaluate(int_test_ds)[1]:.3f}')
