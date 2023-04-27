from transformers import pipeline


def bert(question, evidence):
    bert_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad",
                          tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")
    answer = bert_model({"question": question, "context": evidence})
    return answer["answer"]


while True:
    question = input("Please enter your question:")
    evidence = input("Please enter the evidence:")

    print("Answer:", bert(question, evidence))