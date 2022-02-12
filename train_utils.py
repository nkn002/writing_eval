import torch
from tqdm import tqdm
from config import Config
from sklearn.metrics import accuracy_score

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

config = Config()

def train_one_epoch(model, train_data, optimizer):
    avg_loss = 0
    tr_accuracy = 0
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    model.train()
    for idx, batch in enumerate(tqdm(train_data)):
        ids = batch['input_ids'].to(config.device, dtype = torch.long)
        mask = batch['attention_masks'].to(config.device, dtype = torch.long)
        labels = batch['labels'].to(config.device, dtype = torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                               return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        #tr_labels.extend(labels)
        #tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config.max_grad_norm
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    
    return epoch_loss, tr_accuracy



def val_one_epoch(model, val_data):
    val_loss, val_accuracy = 0, 0
    nb_val_examples, nb_val_steps = 0, 0
    model.eval()
    for idx, batch in enumerate(val_data):
        ids = batch['input_ids'].to(config.device, dtype = torch.long)
        mask = batch['attention_masks'].to(config.device, dtype = torch.long)
        labels = batch['labels'].to(config.device, dtype = torch.long)
        
        with torch.no_grad():
            loss, val_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                   return_dict=False)
        val_loss += loss.item()

        nb_val_steps += 1
        nb_val_examples += labels.size(0)
        
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = val_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        #tr_labels.extend(labels)
        #tr_preds.extend(predictions)

        tmp_val_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        val_accuracy += tmp_val_accuracy
        
    val_loss = val_loss / nb_val_steps
    val_accuracy = val_accuracy / nb_val_steps
    
    return val_loss, val_accuracy