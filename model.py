from transformers import BertPreTrainedModel, BertModel
from torch import nn
import csv
import torch
import warnings
warnings.filterwarnings('ignore')


def load(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as file:
        # read the file and remove the header line
        reader = csv.reader(file, delimiter='\t', quotechar='"')
        next(reader, None)
        for data in reader:
            tmp = {'question': data[1], 'answer': data[5], 'label': int(data[6])}
            result.append(tmp)
    return result


def padding(seq, max_len, pad_fig=0):
    # transform the sequence into fixed length
    pad_len = max_len - len(seq)
    seq += [pad_fig] * pad_len
    return seq


def tokenize(data, max_len, tokenizer, device):
    res = []
    for tuples in data:
        # use encode_plus() method to encode
        encode_res = tokenizer.encode_plus(tuples['question'], tuples['answer'], max_length=max_len, add_special_tokens=True, trunction=True)
        input_ids, token_type_ids = encode_res["input_ids"], encode_res["token_type_ids"]
        input_ids = padding(input_ids, max_len)  # give the padding
        token_type_ids = padding(token_type_ids, max_len)

        # initialise the attention mask with element 1
        attention_mask = [1] * len(input_ids)
        attention_mask = padding(attention_mask, max_len)  # padding to the same size as input sequence
        label = tuples['label']
        res.append((input_ids, attention_mask, token_type_ids, label))

    total_input_ids = []
    total_attention_mask = []
    total_token_type_ids = []
    total_labels = []

    for element in res:
        total_input_ids.append(torch.tensor(element[0], dtype=torch.int64, device=device))
        total_attention_mask.append(torch.tensor(element[1], dtype=torch.int64, device=device))
        total_token_type_ids.append(torch.tensor(element[2], dtype=torch.int64, device=device))
        total_labels.append(torch.tensor(element[3], dtype=torch.int64, device=device))

    total_input_ids = torch.stack(total_input_ids)
    total_attention_mask = torch.stack(total_attention_mask)
    total_token_type_ids = torch.stack(total_token_type_ids)
    total_labels = torch.stack(total_labels)
    return torch.utils.data.TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_labels)


class BertQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQA, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # only fine-tuning the parameters in the layers after bert
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.result = nn.Linear(config.hidden_size, 2)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        # extract features using bert
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        # convert features of each data into the scores of two labels
        split = outputs[0][:, 0, :]
        score = self.result(split).squeeze()
        # use softmax function to handle the scores and pick the larger one
        predicted_labels = nn.functional.softmax(score, dim=-1)

        if labels is not None:  # training process
            loss = self.loss(predicted_labels, labels)
            return loss, predicted_labels
        else:  # testing process
            return predicted_labels




