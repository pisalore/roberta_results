from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr

"""
Evaluation for STS task. The metric it the Pearson's Correlation Coefficient.
Regression problem.
"""
roberta = RobertaModel.from_pretrained(
    '/home/lpisaneschi/ml/fairseq/checkpoints/',
    checkpoint_file='checkpoint_best_stsb_base.pt',
    data_name_or_path='../STS-B-bin'
)
roberta.cuda()
roberta.eval()
gold, pred = [], []
with open('../glue_data/STS-B/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
        tokens = roberta.encode(sent1, sent2)
        features = roberta.extract_features(tokens)
        predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
        gold.append(target)
        pred.append(predictions.item())
        print(target, predictions.item(), sent1, sent2)

print('| Pearson: ', pearsonr(gold, pred))