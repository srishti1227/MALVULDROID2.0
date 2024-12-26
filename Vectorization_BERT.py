from tokenize import tokenize

from transformers import BertTokenizer, BertModel
import torch

from Preprocessing import malware_df, vulnerability_df

#Load prretrained Bert model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#set the model to evalutation mode
model.eval()

#BERT Model works with TokenIDs, so in order to generate the TokenIDs we must tokenize the description
def tokenize_descriptions(descriptions):
    inputs = tokenizer(descriptions.tolist(), padding=True,truncation=True, max_length=512, return_tensors='pt')
    return inputs

#Tokenize malware and vulnerability descriptions
malware_inputs  = tokenize_descriptions(malware_df['description'])
vulnerability_inputs = tokenize_descriptions(vulnerability_df['description'])

@torch.no_grad()
def get_word_embeddings(inputs):
    outputs = model(**inputs)
    last_hidden_states =outputs.last_hidden_state
    cls_embeddings = last_hidden_states[:,0,:]
    return cls_embeddings

#get malware and Vulnerability word embeddings
malware_embeddings = get_word_embeddings(malware_inputs)
vulnerability_embeddings = get_word_embeddings(vulnerability_inputs)

print("These are malware embeddings")
print(malware_embeddings.shape)
print("These are Vulnerability embeddings")
print(vulnerability_embeddings.shape)

