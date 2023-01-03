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
