# %% markdown
# # Fine-tuning BERT on long texts
# %% markdown
# In this notebook we explore different approach to overcome one of the main limitation of BERT (which stands for Bidirectional Encoder Representations from Transformers), the ability to process long document. In fact BERT can only be applied on text that have less than 512 token after tokenization with the Bert Tokenizer.
#
#
# &nbsp;
#
#
# We will implement [this paper, which introduce a new method to deal with Long Documents : RoBERT (Recurrence over BERT)](https://arxiv.org/abs/1910.10781).
#
#
#
# We will also implement [this paper, which introduce diferents methods to deal with Long Documents and BERT](https://arxiv.org/abs/1905.05583) to see if the RoBERT paper bring some significative improvement on the classification of Long Texts with BERT.
#
# This paper introduce 2 main approaches, the **Truncation methods** and the **Hierarchical methods** :
#  * Truncation methods
#    * head-only
#    * tail-only
#    * head+tail
#  * Hierarchical methods
#    * mean pooling
#    * max pooling
#
# The Truncation methods applies to the input of the BERT model (the Tokens), while the Hierarchical methods applies to the ouputs of the Bert model (the embbeding), we will go into more detail in the respective parts
#
#
# &nbsp;
#
#
# The goal of the original RoBERT article was to solve the following problem: BERT has a fixed input token count; how can we use its power on long texts. This notebook implements the same approach using HuggingFace's `transformers` library and `pytorch`.
#
# The dataset used is the *US Consumer Finance Complaints* available on [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints).
#
# Basically, the article goes as follows:
# 1. Read the data and do some basic preprocessing
# 2. Break the documents into smaller segments with a number of tokens that can be handled by BERT
# 3. Fine-tune BERT on those segments using a classification headremain[:idx]
# 4. Combine the segments of each document by using an LSTM. The fixed output size of the LSTM can be used by a single fully connected layer for the final classification.
#
#
# &nbsp;
#
#
# For code clarity, we separated out some parts of the code into python scripts that are fully commented. They will be referenced throughout the notebook.
#
# Here is a graph that represents the differents Interaction between the differents Classes :
# %% markdown
# ![img/Class_Interactions.png](img/Class_Interactions.png)
# %% codecell
# %matplotlib inline" ".join(sec['pars'])
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW# get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import time

from utils import *
from Custom_Dataset_Class import ConsumerComplaintsDataset1
from Bert_Classification import Bert_Classification_Model
from RoBERT import RoBERT_Model

from BERT_Hierarchical import BERT_Hierarchical_Model
import warnings
warnings.filterwarnings("ignore")
# %% codecell
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
# %% markdown
# ## Data Exploration :
# %% markdown
# The dataset used in this work was retrieved from kaggle as said in the paper, these are consumer complaints about financial products and services which are sent by the CFPB (Consumer FinancialProtection  Bureau)  to  the  company  for  answer.
#
# The  dataset  consists  of  555957  rows  and  18columns.
#
# As our model attempts to predict which product the complaint is about, we only used the consumer-complaint-narrative and product columns.
# * comsumer-complaint-narrative:  contains the consumer complaint in text format.
# * product:  label of the product concerned by the complaint\
#
# The final dataset used in this work consists of 555957 rows and 2 columns (one column for the texts and the other for the labels).
# %% codecell
# Load the dataset into a pandas dataframe.
filename = '/fbf/fbf_repos/feedbackfruits-rnd-data-office/data/section-classification/outputs/covid_sections'
df=pd.read_csv(filename)

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

train_raw = df[df.text.notnull()]
print('Number of training sentences with text not null: {:,}\n'.format(train_raw.shape[0]))

# Display 10 random rows from the data.
train_raw.sample(10)
train_raw.text.apply(lambda x: len(x.split()) if len(x.split())<800 else 800).plot(kind='hist', title="nombre d'ocurence par nombre de mots dans un commentaire")
train_raw['len_txt'] =train_raw.text.apply(lambda x: len(x.split()))
train_raw.describe()
#Select only the row with number of words greater than 250:
train_raw = train_raw[train_raw.len_txt >249]
train_raw.shape
#Select only the column 'consumer_complaint_narrative' and 'product'
train_raw.reset_index(inplace=True, drop=True)
train_raw.head()
# %% codecell

# %% markdown
# ## Data preprocessing and segmentation:
# %% markdown
# ### 1. Preprocessing:
# The preprocessing step goes as follows:
# 1. Remove all documents with fewer than 250 tokens. We want to concentrate only on long texts
# 2. Consolidate the classes by combining those that are similar. (e.g.: "Credit card" or "prepaid card" complaints) :
#  * Credit reporting‘ to ‘Credit reporting, credit repair services, or other personal consum
#  * ‘Credit card‘ to ‘Credit card or prepaid card‘.
#  * ‘Payday loan‘ to ‘Payday loan, title loan or personal loan‘1
#  * ‘Virtual currency‘ to ‘Money transfer, virtual currency or money servic
# 3. Remove all non-word characters
# 4. Encode the labels
# 5. Split the dataset in train set (80%) and validation set (20%).
#
# ### 2. Segmentation and tokenization:
# First, each complaint is split into 200-token chunk with an overlap of 50 between each of them. This means that the last 50 tokens of a segment are the first 50 of the next segment.
# Then, each segment is tokenized using BERT's tokenizer. This is needed for two main reasons:
# 1. BERT's vocabulary is not made of just English words, but also subwords and single characters
# 2. BERT does not take raw string as inputs. It needs:
#     - token ids: those values allow BERT to retrieve the tensor representation of a token
#     - input mask: a tensor of 0s and 1s that shows whether a token should be ignored (0) or not (1) by BERT
#     - segment ids: those are used to tell BERT what tokens form the first sentence and the second sentence (in the next sentence prediction task)
#
# The parameter `MAX_SEQ_LENGTH` ensures that any tokenized sentence longer that that number (200 in this case) will be truncated.
# Each returned segment is given the same class as the document containing it.
#
# PyTorch offers the `Dataset` and `DataLoader` classes that make it easier to group the data reading and preparation operations while decreasing the memory usage in case the dataset is large.
# We implemented the above steps in the `Custom_Dataset_Class.py` file. Our dataset class `ConsumerComplaintsDataset1` has a constructor (`__init__`) taking the necessary parameters to load the .csv file, segment the documents, and then tokenize them. It also preprocesses the data as explained in the preprocessing section.
# The two other important methods are the following:
# - `__len__` returns the number of documents
# - `__getitem__` is the method where most of the work is done. It takes a tensor of idx values and returns the tokenized data:
#
#
# &nbsp;
#
#
# As said above, there is differents approaches for overflowing tokens, we added the differents strategies (described in [the following paper](https://arxiv.org/abs/1905.05583)) via the use of the parameter `approach` in the initialisation of Consumer Complaints Dataset class, here is the differents value of the `approach` parameter to handle that situation :
# - **all**: overflowing tokens from a document are used to create new 200 token chunk with 50 tokens overlap between them
# - **head**: overflowing tokens are truncated. Only the first 200 tokens are used.
# - **tail**: only the last 200 tokens are kept.
# %% codecell
TRAIN_BATCH_SIZE=8
EPOCH=3
validation_split = .2
shuffle_dataset = True
random_seed= 42
MIN_LEN=249
MAX_LEN = 100000
CHUNK_LEN=200
OVERLAP_LEN=50
#MAX_LEN=10000000
#MAX_SIZE_DATASET=1000

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset=ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)


#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=my_collate1)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)
# %% markdown
# ### 3. Fine-tuning on the 200 tokens chunks:
# BERT is fine-tuned on the 200 tokens chunks
#
# In our implementation, we put a neural network on top of the pooled output from BERT, each BERT's input token has a embeding as an output, the embedding of the `CLS` token (the first token) corresponds to the pooled output of the all sentence input (the 200 tokens chunk in our case).
#
# The neural network is composed of a dense layer with a SoftMax activation function.
#
# This Model corresponds to the class `Bert_Classification_Model` defined in the file `Bert_Classification.py`. This class inherits from `torch.nn.Module` which is the parent of all neural network models.
#
# Then, in the function `train_loop_fun1` defined in `utils.py`, for each batch, the list of dictionaries containing the values for the token_ids, masks, token_type_ids, and targets are respectively concatenated into `torch.tensors` which are then fed into the model in order to get predictions and apply backpropagation according to the Cross Entropy loss.
# %% markdown
# #### First segmetation approach: all
# %% markdown
# in this approach, we will consider each chunk of 200 tokens as a new document, so if a document is split into 3 chunk of 200, tokens we will consider each chunks as a new document with the same label.\
# We use this approach to fine tune BERT, so this model will be used as an input for RoBERT.
# %% markdown
# ![img/each_Chunk_as_Document.png](img/each_Chunk_as_Document.png)
# %% codecell
device="cpu"
lr=3e-5#1e-3
num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

model=Bert_Classification_Model().to(device)
optimizer=AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
for epoch in range(EPOCH):
    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=train_loop_fun1(train_data_loader, model, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=eval_loop_fun1(valid_data_loader, model, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print("\t§§ model has been saved §§")
    torch.save(model, f"model1/model_epoch{epoch+1}.pt")
# %% codecell
pd.DataFrame(np.array([[np.mean(x) for x in batches_losses], [np.mean(x) for x in val_losses]]).T,
                   columns=['Training', 'Validation']).plot(title="loss")
# %% codecell
pd.DataFrame(np.array(val_acc).T,
                   columns=['Validation']).plot(title="accuracy")
# %% codecell

# %% markdown
# ##### Now we will experience the [Truncation strategies presented in this paper](https://arxiv.org/abs/1905.05583)
#
# such as:
# * Tunction Method:
#     + Head only
#     + Tail only
# * Hierarchical Method:
#     + Mean pooling
#     + Max pooling

# %% markdown
# the Hierarchical methods applies to the ouputs of the Bert model (the embbeding)
# The input text is firstly divided into k = L/510 fractions, which is fed into BERT to obtain the representation of the k text fractions. The representation of each fraction is the hidden state of the `[CLS]` tokens of the last layer. Then we use mean pooling, max pooling
#
# %% markdown
# #### Mean Pooling
# %% markdown
# in this approach, we average the embedding of all k chunks across each dimensions.
# %% markdown
# ![img/Mean_Pooling_Hierarchical.png](img/Mean_Pooling_Hierarchical.png)
# %% codecell
TRAIN_BATCH_SIZE=3
EPOCH=1
validation_split = .2
shuffle_dataset = True
random_seed= 42
MIN_LEN=249
MAX_LEN = 100000
CHUNK_LEN=200
OVERLAP_LEN=50
#MAX_LEN=10000000
#MAX_SIZE_DATASET=1000

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset=ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)


#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=my_collate1)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)


device="cpu"
lr=3e-5#1e-3
num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

pooling_method="mean"
model_hierarchical=BERT_Hierarchical_Model(pooling_method=pooling_method).to(device)
optimizer=AdamW(model_hierarchical.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
for epoch in range(EPOCH):
    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=rnn_train_loop_fun1(train_data_loader, model_hierarchical, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=rnn_eval_loop_fun1(valid_data_loader, model_hierarchical, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print(f"\t§§ the Hierarchical {pooling_method} pooling model has been saved §§")
    torch.save(model_hierarchical, f"model_hierarchical/{pooling_method}_pooling/model_{pooling_method}_pooling_epoch{epoch+1}.pt")
# %% codecell

# %% markdown
# #### Max Pooling
# %% markdown
# in this approach, we take the maximum embedding of all the k chunks across each dimensions.
# %% markdown
# ![img/Max_Pooling_Hierarchical.png](img/Max_Pooling_Hierarchical.png)
# %% codecell
TRAIN_BATCH_SIZE=3
EPOCH=1
validation_split = .2
shuffle_dataset = True
random_seed= 42
MIN_LEN=249
MAX_LEN = 100000
CHUNK_LEN=200
OVERLAP_LEN=50
#MAX_LEN=10000000
#MAX_SIZE_DATASET=1000

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset=ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)


#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=my_collate1)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)


device="cpu"
lr=3e-5#1e-3
num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

pooling_method="max"
model_hierarchical=BERT_Hierarchical_Model(pooling_method=pooling_method).to(device)
optimizer=AdamW(model_hierarchical.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
for epoch in range(EPOCH):
    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=rnn_train_loop_fun1(train_data_loader, model_hierarchical, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=rnn_eval_loop_fun1(valid_data_loader, model_hierarchical, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print(f"\t§§ the Hierarchical {pooling_method} pooling model has been saved §§")
    torch.save(model_hierarchical, f"model_hierarchical/{pooling_method}_pooling/model_{pooling_method}_pooling_epoch{epoch+1}.pt")
# %% codecell

# %% markdown
# # RoBERT RNN classifier on top of the Fine Tuned Bert Model
# %% markdown
#  The input text is firstly divided into k = L/510 fractions, which is fed into the fine tuned BERT (as describe above, *cf: First segmetation approach: all*) to obtain the representation of the k text chunks. The representation of each fraction is the hidden state of the `[CLS]` tokens of the last layer.
#
# Each chunk embedding (representation) become the input of an LSTM cell, this way the order is preserved and the length of the document is not a limitation anymore because of the dynamic aspect of the LSTM that allow different variable sequence lengths (accros different batches)
#
# we then pass the last hidden state (nbDoc * 100) to a neural network with the same architecture (as describe above, we use the same neural network architecture for classification through the all notebook, [cf: 3. Fine-tuning on the 200 tokens chunks](#3.-Fine-tuning-on-the-200-tokens-chunks:))
# %% markdown
# ![img/RoBERT.png](img/RoBERT.png)
# %% codecell
TRAIN_BATCH_SIZE=3
EPOCH=3
validation_split = .2
shuffle_dataset = True
random_seed= 42
MIN_LEN=249
MAX_LEN = 100000
CHUNK_LEN=200
OVERLAP_LEN=50
#MAX_LEN=10000000
#MAX_SIZE_DATASET=1000

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset=ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)


#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=my_collate1)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)


device="cpu"
lr=3e-5#1e-3
num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

model=torch.load("model1/model_epoch2.pt")

model_rnn=RoBERT_Model(bertFineTuned=list(model.children())[0]).to(device)
optimizer=AdamW(model_rnn.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0,
                                        num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
for epoch in range(EPOCH):
    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp=rnn_train_loop_fun1(train_data_loader, model_rnn, optimizer, device)
    epoch_loss=np.mean(batches_losses_tmp)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1=time.time()
    output, target, val_losses_tmp=rnn_eval_loop_fun1(valid_data_loader, model_rnn, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate=evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print("\t§§ the RNN model has been saved §§")
    torch.save(model_rnn, f"model_rnn1/model_rnn_epoch{epoch+1}.pt")
# %% codecell
pd.DataFrame(np.array([[np.mean(x) for x in batches_losses], [np.mean(x) for x in val_losses]]).T,
                   columns=['Training', 'Validation']).plot(title="loss")
# %% codecell
pd.DataFrame(np.array(val_acc).T,
                   columns=['Validation']).plot(title="accuracy")
# %% codecell

# %% markdown
# # Summary
# %% codecell
summary=pd.DataFrame({"all":[0.74, 0.57, 0.83], "head only":[0.62, 0.45, 0.87], "tail only":[0.93, 0.75, 0.77], "mean pooling":[0.62, 0.47, 0.87], "max pooling":[0.65, 0.45, 0.87], "RoBERT":[0.46, 0.34, 0.91]}, index=["avg_loss_train", "avg_loss_val", "accuracy"])
summary.columns.name = 'after one epoch'
summary.style.set_properties(
    subset=['RoBERT'],
    **{'font-weight': 'bold'}
)
# %% markdown
# We can observe that the RoBERT Model give the best result, so we can conclude that this Model a net State Of the Art improvement in term of the loss function and the accuracy, the second model is the head only, this make sense because we can imagine that the consumer introduce his complain within the first part of his comment.
# %% codecell
