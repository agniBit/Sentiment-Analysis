from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import dataloaders
from torch.nn.utils import clip_grad_norm_
import random
import time
import torch
from utils import *
import dataset
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
# Use the 12-layer BERT model with binary class
class Model:
    def __init__(self):
        # check device CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # define bert model with 2 classes and push to device memory
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        ).to(self.device)
        # define dataloader for data stream in training
        self.dataloaders = dataloaders.Dataloaders()
        # max input sequence length
        self.MAX_LEN = 46

    def train(self,epochs = 4):
        train_dataloader, val_dataloader = self.dataloaders.get_dataloaders()
        # use AdamW optimizer for training
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        # seed torch and random so we can generate shame result latter
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        loss_values = []
        for epoch_i in range(0, epochs):
            # store starting time
            start = time.time()
            total_loss = 0
            # set model in training model as model behave different in evaluation and training mode
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - start)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'
                          .format(step, len(train_dataloader), elapsed))
                # get ids , mask and labels from dataloader
                input_ids = batch[0].to(self.device)
                input_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                # clear gradients
                self.model.zero_grad()
                # feed froward model
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask,
                                labels=labels)
                loss = outputs[0]
                # accumulate loss
                total_loss += loss.item()
                loss.backward()
                # clip the gradient preventing gradient exploding
                clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("\n\n val")
            start = time.time()
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in val_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, labels = batch
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                # calculate accuracy
                tmp_eval_accuracy = accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  val took: {:}".format(format_time(time.time() - start)))
            print("")
        print("Training complete!")


    def predict(self,text):
        # load model
        try:
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load('./model.pth'))
            else:
                self.model.load_state_dict(torch.load('./model.pth',map_location='cpu'))
            # set model to evaluation mode
            self.model.eval()
            data = dataset.Dataset()
            # preprocess data
            text = data.preprocess(text)
            # initialise bert tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            input_token_seq = []
            # tokenize text
            encoded_text = tokenizer.encode(str(text), add_special_tokens=True)
            input_token_seq.append(encoded_text)
            input_token_seq = pad_sequences(input_token_seq, maxlen=self.MAX_LEN, dtype="long",
                                            value=0, truncating="post", padding="post")
            attention_masks = []
            # create attention mask for distinguishing padding and tokens
            att_mask = [int(token_id > 0) for token_id in input_token_seq[0]]
            attention_masks.append(att_mask)
            # feed forward without calculating gradients
            with torch.no_grad():
                outputs = self.model(torch.tensor(input_token_seq), token_type_ids=None,
                                     attention_mask=torch.tensor(attention_masks)).logits.detach().cpu().numpy()
            if np.argmax(outputs,axis=1)[0]==1:
                return str('positive')
            else:
                return str('negative')
        except:
            return 'error during predicting'
