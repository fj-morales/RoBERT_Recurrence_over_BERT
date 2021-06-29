import torch
from dis_utils import *
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from DisCustom_Dataset_Class import DisConsumerComplaintsDataset1
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import plot_confusion_matrix


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model_path = "./model_rnn1/model_rnn_epoch3.pt"

model = torch.load(model_path)

TRAIN_BATCH_SIZE=8
EPOCH=3
validation_split = .9
shuffle_dataset = True
random_seed= 42
MIN_LEN=249
MAX_LEN = 100000
CHUNK_LEN=200
OVERLAP_LEN=50


bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
dataset=DisConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)

output, target, val_losses_tmp = rnn_eval_loop_fun1(valid_data_loader, model, device)
tmp_evaluate=evaluate(target.reshape(-1), output)
print(f"=====>\t{tmp_evaluate}")

y_pred = [np.argmax(x) for x in output]

IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier", "classes_": dataset.LE.classes_})
plot_confusion_matrix(IC, dataset.LE.inverse_transform(y_pred), dataset.LE.inverse_transform(target), normalize='true', values_format='.2%')
