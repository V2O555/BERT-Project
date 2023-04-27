import numpy as np

from model import tokenize, load, BertQA
import transformers
import torch
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_path = './models/bert-pretrain/'
config = transformers.BertConfig.from_pretrained(model_path)
model = BertQA.from_pretrained(model_path, config=config)
model.to(device)


batch_size = 128  # batch size
max_length = 256  # max length of a sequence
tokenizer = transformers.BertTokenizer.from_pretrained(model_path,do_lower_case=True)
# load the train dataset
train_file = load('data/WikiQA-train.tsv')
train_dataset = tokenize(train_file, max_length, tokenizer, device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)


epoch_num = 40
learning_rate = 1e-5
adam_eps = 1e-8
# path to save model
save_path = './models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# initialise the optimizer
optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=adam_eps)


loss_list = []
for epoch in range(epoch_num):
    print("Training epoch", epoch+1)
    tmp_loss = []
    for idx, batch in enumerate(train_dataloader):
        model.train()
        model.zero_grad()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
        outputs = model(**inputs)
        loss, results = outputs
        loss.backward()
        optimizer.step()
        tmp_loss.append(loss.item())
    loss_list.append(np.mean(tmp_loss))
# save the model
torch.save(model.state_dict(), os.path.join(save_path, 'best_param.bin'))


plt.plot(loss_list)
plt.show()