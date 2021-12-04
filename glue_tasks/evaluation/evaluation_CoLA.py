import numpy as np
from sklearn.metrics import matthews_corrcoef
from fairseq.models.roberta import RobertaModel

"""
Evaluation for CoLA task. The metric it the Matthews Correlation Coefficient.
"""
roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/roberta_results/checkpoints/',
    checkpoint_file='checkpoint_best_cola_large.pt',
    data_name_or_path='../glue_tasks/CoLA-bin'
)
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
predictions = []
ground_truth = []
roberta.cuda()
roberta.eval()
with open('../glue_tasks/glue_data/CoLA/dev.tsv', encoding='utf-8') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent, target = tokens[3], tokens[1]
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        prediction_label = int(prediction_label)
        print(prediction_label, target, sent)
        target = int(target)
        predictions.append(prediction_label)
        ground_truth.append(target)
        nsamples += 1

print('| Accuracy: ', float(ncorrect)/float(nsamples))
MCC = matthews_corrcoef(ground_truth, predictions)
print('| MCC: ', MCC)
