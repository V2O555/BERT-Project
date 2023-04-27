import numpy as np
import torch

from model import tokenize, load, BertQA
import transformers


def calculate_em_score(y_pred, y_true):
    """
    计算准确匹配（Exact Match，EM）得分

    参数:
    y_true (Tensor): 真实标签张量，形状为 (n_samples,)
    y_pred (Tensor): 预测标签张量，形状为 (n_samples,)

    返回:
    em_score (float): EM得分
    """
    assert len(y_true) == len(y_pred), "y_true 和 y_pred 的长度不一致"

    em_score = 0.0  # 初始化 EM 得分为0

    for i in range(len(y_true)):
        # 使用索引逐个比较 y_true 和 y_pred 中的值，并进行布尔运算
        em_score += y_true[i] == y_pred[i]

    # 计算 EM 得分并返回
    em_score = em_score / len(y_true)

    return em_score
def AP(output, target):
    output = torch.tensor(output,dtype=torch.float)
    target = torch.tensor(target,dtype=torch.float)
    _, indexes = torch.sort(output, descending=True)
    target = target[indexes].round()
    total = 0.
    for i in range(len(output)):
        index = i+1
        if target[i]:
            total += target[:index].sum().item() / index
    # 如果没有正确答案，target.sum = 0 出现异常
        return total/target.sum().item()
    # 没有正确答案返回 0


def MAP(outputs, targets):
    assert(len(outputs) == len(targets))
    res = []
    for i in range(len(outputs)):
        res.append(AP(outputs[i], targets[i]))
    return np.mean(res)


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


