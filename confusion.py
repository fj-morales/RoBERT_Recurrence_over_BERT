import json
from sklearn.metrics import plot_confusion_matrix

label_data = json.load(open('./labels.json', 'rt'))

y_pred = label_data['pred']
true = label_data['true']

IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier", "classes_": ['abs', 'conc', 'int']})
plot_confusion_matrix(IC, y_pred, true,
                #  normalize='true', values_format='.2%'
                )


help(plot_confusion_matrix)

filename = '/fbf/fbf_repos/feedbackfruits-rnd-data-office/data/section-classification/outputs/test_covid_sections.csv'
df = pd.read_csv(file_location, dtype="unicode")
