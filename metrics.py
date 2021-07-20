import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score

def measure_metrics(model, X_test, y_test, y_pred):
    '''

    Generate the confusion matrix of the model

    '''


    f1 = f1_score(y_test, y_pred, average='weighted')
    ac = accuracy_score(y_test, y_pred)


    matrix_classes = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=['negative', 'neutral','positive'],
        columns=['negative', 'neutral','positive']
    )

    print(f"accuracy:  {ac:.2f}", f"F1_score: {f1:.2f}",
    'Confusion_matrix:', matrix_classes, sep="\n")

