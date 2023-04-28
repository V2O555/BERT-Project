import numpy as np
import torch

from model import tokenize, load, BertQA
import transformers


max_length = 256
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = './models/bert-pretrain/'
tokenizer = transformers.BertTokenizer.from_pretrained(model_path,do_lower_case=True)
batch_size = 128

test_file = load('data/WikiQA-test.tsv')
test_dataset = tokenize(test_file,max_length, tokenizer,device)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

config = transformers.BertConfig.from_pretrained(model_path)
model = BertQA.from_pretrained(model_path,config=config)
model.to(device)

model.load_state_dict(torch.load('models/best_param.bin'))


total_correct = 0
total_data = 0
total_predict = []
total_target = []
for step, data in enumerate(test_dataloader):
    model.eval()
    with torch.no_grad():
        inputs = {'input_ids': data[0], 'attention_mask': data[1], 'token_type_ids': data[2]}
        predict = model(**inputs)
        predict = torch.argmax(predict, dim=1)
        correct_data = predict - data[3]
        for element in correct_data:
            if element == 0:
                total_correct += 1
        total_data += len(data[3])
        total_target.append(data[3])
        total_predict.append(predict)
print("Accuracy is", float(total_correct) / total_data)


