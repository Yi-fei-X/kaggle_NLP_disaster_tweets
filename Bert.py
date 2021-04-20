import numpy as np
import pandas as pd
import random
import re
import csv
import torch
from prettytable import PrettyTable
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Read the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#add keyword to text
train_df['keyword'] = train_df['keyword'].fillna('nokeyword')
train_df['text'] = train_df['keyword'] + ' ' + train_df['text']
test_df['keyword'] = test_df['keyword'].fillna('nokeyword')
test_df['text'] = test_df['keyword'] + ' ' + test_df['text']


#Clean the data
def cleandata(text):
    text = text.lower()
    #Remove webpage
    text = re.sub(r'http://\S+', '', text)
    text = re.sub(r'https://\S+', '', text)
    text = re.sub(r'http', '', text)
    #Remove mention
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'via', '', text)
    #Remove some sign
    text = re.sub(r'#', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'[*]', '', text)
    text = re.sub(r';\)', '', text)
    text = re.sub(r':\)', '', text)
    text = re.sub(r'-', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r';', '', text)
    text = re.sub(r'<', '', text)
    text = re.sub(r'=', '', text)
    text = re.sub(r'>', '', text)
    text = re.sub('\+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\|', '', text)
    text = re.sub('\[', '', text)
    text = re.sub('\]', '', text)
    text = re.sub('\(', '', text)
    text = re.sub('\)', '', text)
    #Remove redundant sign
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub('\s+', ' ', text).strip()

    text = re.sub('nokeyword ', '', text)
    #Remove non-ascii
    text = text.encode("ascii", errors="ignore").decode()
    return text

train_df['text'] = train_df['text'].apply(cleandata)
test_df['text'] = test_df['text'].apply(cleandata)

#load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#find the maximum length of our dataset
max_len = 0
for sentence in train_df["text"]:
    input_id = tokenizer.encode(sentence, add_special_tokens=True)
    max_len = max(max_len, len(input_id))
for sentence in test_df["text"]:
    input_id = tokenizer.encode(sentence, add_special_tokens=True)
    max_len = max(max_len, len(input_id))
print("maximum length is ", max_len)

#tokenize text
input_ids = []
attention_masks = []
for sentence in train_df["text"]:
    train_df_encode = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt')
    input_ids.append(train_df_encode["input_ids"])
    attention_masks.append(train_df_encode["attention_mask"])

input_ids_test = []
attention_masks_test = []
for sentence in test_df["text"]:
    test_df_encode = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt')
    input_ids_test.append(test_df_encode["input_ids"])
    attention_masks_test.append(test_df_encode["attention_mask"])

#convert to tensor
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
targets = torch.Tensor(train_df['target'])
targets = targets.long()

input_ids_test = torch.cat(input_ids_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)

#load data
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, targets)
testset = torch.utils.data.TensorDataset(input_ids_test, attention_masks_test)

#Train val split
train_len = int(len(dataset)*0.8)   #select 80% training 20% validation
val_len = len(dataset) - train_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

#build dataloader
batchsize = 16
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)
val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=batchsize, shuffle=False)
test_set_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

#Model: BertForSequenceClassification (This idea is from "BERT Fine-Tuning Tutorial with PyTorch")
epochs = 2

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2)
model.cuda()
optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)

#training
train_loss_total = []
val_loss_total = []
val_acc_total = []
for epoch in range(epochs):
    train_loss = 0
    model.train()
    #train on training set
    for i, data in enumerate(train_set_loader):
        input_id = data[0].to(device)
        attention_mask = data[1].to(device)
        target = data[2].to(device)
        model.zero_grad()
        loss, logits = model(input_id, token_type_ids=None, attention_mask=attention_mask, labels=target)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_loss_total.append(train_loss/i)
    print('BERT: [%d] train loss: %.5f' %
          (epoch + 1, train_loss/i))
    train_loss = 0

    #test on validation set
    val_loss = 0
    val_accuracy = 0
    val_correct = 0
    val_total = 0
    model.eval()
    for i, data in enumerate(val_set_loader):
        input_id = data[0].to(device)
        attention_mask = data[1].to(device)
        target = data[2].to(device)
        with torch.no_grad():
            loss, logits = model(input_id,token_type_ids=None, attention_mask=attention_mask, labels=target)
        val_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label = target.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        label_flat = label.flatten()
        val_correct += np.sum(pred_flat == label_flat)
        val_total += len(label_flat)

    val_loss_total.append(val_loss/i)
    print('BERT: [%d] val loss: %.5f' %
          (epoch + 1, val_loss/i))
    val_loss = 0

    val_accuracy = val_correct/val_total
    val_acc_total.append(val_accuracy)
    print("validation accuracy is ", val_accuracy)

#Form a table
Table_BERT = PrettyTable()
Table_BERT_title = (np.arange(epochs) + 1).tolist()
Table_BERT_title.insert(0,'number of epochs')
Table_BERT.field_names = Table_BERT_title

Table_BERT_train = train_loss_total.copy()
Table_BERT_train.insert(0,"train_loss_total")
Table_BERT.add_row(Table_BERT_train)

Table_BERT_val = val_loss_total.copy()
Table_BERT_val.insert(0,"val_loss_total")
Table_BERT.add_row(Table_BERT_val)

Table_BERT_acc = val_acc_total.copy()
Table_BERT_acc.insert(0,"val_acc_total")
Table_BERT.add_row(Table_BERT_acc)
print(Table_BERT)

print()

#train on all data
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2)
model.cuda()
optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)

train_all_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

train_loss_total = []
for epoch in range(epochs):
    train_loss = 0
    model.train()
    #train on training set
    for i, data in enumerate(train_all_loader):
        input_id = data[0].to(device)
        attention_mask = data[1].to(device)
        target = data[2].to(device)
        model.zero_grad()
        loss, logits = model(input_id, token_type_ids=None, attention_mask=attention_mask, labels=target)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_loss_total.append(train_loss/i)
    print('BERT: [%d] train loss: %.5f' %
          (epoch + 1, train_loss/i))
    train_loss = 0

#predict test
test_pred = []
prediction = []
model.eval()
for _, data in enumerate(test_set_loader):
    input_id = data[0].to(device)
    attention_mask = data[1].to(device)
    with torch.no_grad():
        output = model(input_id, token_type_ids=None, attention_mask=attention_mask)
    logits = output[0]
    logits = logits.detach().cpu().numpy()
    test_pred = np.argmax(logits, axis=1).flatten()
    prediction = np.concatenate((prediction, test_pred))
prediction = prediction.astype(np.int64)
#Write my prediction into CSV file
len_test = len(test_df)
index_test = []
for i in range(len_test):
    index_test.append(test_df['id'][i])

with open("labels.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'target'])
    for i in range(len_test):
        writer.writerow([index_test[i], prediction[i]])

print()
