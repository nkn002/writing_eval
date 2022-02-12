from torch.utils.data import Dataset
import torch
import pandas as pd
from config import Config


output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

config = Config()
class WritingEvalDataset(Dataset):
    def __init__(self, df, tokenizer, is_train = True):
        self.data = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        
    def __getitem__(self, index):
        text = self.data.text.values[index]
        if self.is_train:
            entities = self.data.entities.values[index]
        
        encoded = self.tokenizer(text.split(),
                                is_split_into_words=True,
                                padding='max_length',
                                truncation=True,
                                max_length=config.max_length)
        
        input_ids = encoded['input_ids']
        masks = encoded['attention_mask']
        word_ids = [i if i != None else -1 for i in encoded.word_ids()]
        
        if self.is_train:
            prev_id = 0
            labels = []
            for i, word_id in enumerate(word_ids):
                if word_id == -1:
                    labels.append(-100)
                else:
                    label = labels_to_ids[entities[int(word_id)]]
                    labels.append(label)
            
            item = {'input_ids': torch.tensor(input_ids),
                   'attention_masks': torch.tensor(masks),
                   'labels': torch.tensor(labels)}
        else:
            item = {'input_ids': torch.tensor(input_ids),
                   'attention_masks': torch.tensor(masks),
                   'word_ids': torch.tensor(word_ids)}
        return item
    
    def __len__(self):
        return len(self.data)
