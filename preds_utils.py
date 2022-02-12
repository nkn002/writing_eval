import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config


output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

config = Config()
# from https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633
def get_predictions(model, test_data):
    model.eval()
    all_preds = []
    for i, batch in enumerate(tqdm(test_data)):
        input_ids = batch['input_ids'].to(config.device)
        attention_masks = batch['attention_masks'].to(config.device)
        word_ids = batch['word_ids']
        output = model(input_ids, attention_mask = attention_masks)

        pred = torch.argmax(output.logits, axis=-1).detach().cpu().numpy()

        batch_preds = []
        
        #Remove labels if different tokens for same word (subwords)
        for k, text_preds in enumerate(pred):
            prediction = []
            token_preds = [ids_to_labels[i] for i in text_preds]
            word_id = word_ids[k]
            previous_word_idx = -1
            for idx,word_idx in enumerate(word_id):                            
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:              
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            batch_preds.append(prediction)

        all_preds.extend(batch_preds)

    return all_preds
          
# from https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633    
def get_preds_df(val_df, all_preds, threshold=7):
    preds_pre_df = []
    for i in range(len(val_df)):
        #Get the id of 
        idx = val_df.id.values[i]
        pred = all_preds[i]

        preds = list()
        j = 0
        while j < len(pred):
            clfs = pred[j]

            if clfs == 'O':
                j += 1
            else:
                clfs = clfs.replace('B','I')
                end_of_cls = j+1
                
                while end_of_cls < len(pred) and pred[end_of_cls] == clfs:
                    end_of_cls += 1
                    
                if clfs != 'O' and clfs != '' and end_of_cls - j > threshold:
                    pred_list = [str(i) for i in range(j, end_of_cls)]
                    pred_string = ' '.join(pred_list)
                    pred_class = clfs.replace("I-", '')
                    preds.append([idx, pred_class, pred_string])
                j = end_of_cls

        preds_pre_df.extend(preds)
    preds_df = pd.DataFrame(preds_pre_df)
    preds_df.columns = ['id','class','predictionstring']
    return preds_df