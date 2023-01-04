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

# 첫번째 배치 크기와 dtype 출력
for ip, tg in train_ds:
  print("inputs.shape",ip.shape)
  print("inputs.dtype", ip.dtype)
  print("targets.shape", tg.shape)
  print("targets.dtype",  tg.dtype)
  print("inputs[0]:", ip[0])
  print("targets[0]:", tg[0])
  break

# 데이터 전처리
## TextVectorization 층
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

text_vec = TextVectorization(
    max_tokens = 20000,
    output_mode = "multi_hot",
)
t_only_train_ds = train_ds.map(lambda x, y: x)
text_vec.adapt(t_only_train_ds)
binary1_val_ds = train_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
)
binary1_val_ds = val_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
)
binary1_val_ds = test_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
)

# 바이너리 유니그램 데이터셋 출력 확인
for ip, tg in binary1_train_ds:
  print("inputs.shape",ip.shape)
  print("inputs.dtype", ip.dtype)
  print("targets.shape", tg.shape)
  print("targets.dtype",  tg.dtype)
  print("inputs[0]:", ip[0])
  print("targets[0]:", tg[0])
  break
  
# 모델생성 유틸리티
from tensorflow import keras
from tensorflow.keras import layers

# 모델생성 함수 정의
def get_model(max_tokens = 20000, hidden_dim = 16):
  inputs = keras.Input(shape = (max_tokens,))
  x = layers.Dense(hidden_dim, activation = 'relu')(inputs)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(1, activation = 'sigmoid')(x)
  model = keras.Model(inputs, outputs)
  model.compile(optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
  return model

# 훈련  및 테스트
model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary1_gram.keras", save_best_only = True)
]
model.fit(binary1_train_ds.cache(),
          validation_data = binary1_val_ds.cache(),
          epochs = 10,
          callbacks = callbacks)
model = keras.models.load_model("binary1_gram.keras")
print(f' 테스트 정확도: {model.evaluate(binary1_test_ds)[1]:.3f}')

# 바이그램을 반환하는 TextVectorization 층 
text_vec = TextVectorization(
    ngrams=2, 
    max_tokens = 20000,
    output_mode = "multi_hot",
)

# 이진 바이그램 모델 훈련  및 테스트
text_vec.adapt(t_only_train_ds)

binary2_train_ds = train_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
) 
binary2_val_ds = val_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
) 
binary2_test_ds = test_ds.map(
    lambda x, y: (text_vec(x), y),
    num_parallel_calls = 4
) 

model_2 = get_model()
model_2.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras",
                                    save_best_only = True)
]
# 학습
model_2.fit(binary2_train_ds.cache(),
            validation_data = binary2_val_ds.cache(),
            epochs = 10,
            callbacks = callbacks)
model_2 = keras.models.load_model("binary_2gram.keras")
print(f"테스트 정확도: {model.evaluate(binary2_test_ds)[1]:.3f}")

# tfidf
tfidf_text_vec = TextVectorization(
    ngrams = 2, 
    max_tokens = 20000,
    output_mode = 'tf_idf',
)

# tf_idf 모델 훈련  및 테스트
tfidf_text_vec.adapt(t_only_train_ds)
tf2_train_ds = train_ds.map(
    lambda x, y: (tfidf_text_vec(x), y),
    num_parallel_calls = 4
) 
tf2_val_ds = val_ds.map(
    lambda x, y: (tfidf_text_vec(x), y),
    num_parallel_calls = 4
) 
tf2_test_ds = test_ds.map(
    lambda x, y: (tfidf_text_vec(x), y),
    num_parallel_calls = 4)

# 모델 생성
model_tf = get_model()
model_tf.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
                                    save_best_only = True)
]
# 학습
model_tf.fit(tf2_train_ds.cache(),
            validation_data = tf2_val_ds.cache(),
            epochs = 10,
            callbacks = callbacks)
model_tf = keras.models.load_model("tfidf_2gram.keras")
print(f"테스트 정확도: {model.evaluate(tf2_test_ds)[1]:.3f}")

# 원시 문자열 처리 모델 내보내기
imnport tensorflow as tf

inputs = keras.Input(shape = (1,), dtype = "string")
processed_inputs = tfidf_text_vec(inputs)
outputs = model_tf(processed_inputs)
inference_model = keras.Model(inputs, outputs)

raw_text = tf.convert_to_tensor([
    ["That was an excellent movie, I loved it."],
])
pred = inference_model(raw_text)
print(f"긍정적인 리뷰일 확률 : {float(pred[0]*100):.2f} %")
