from sklearn.model_selection import GroupShuffleSplit
from preprocess import preprocess
from preprocess import encode_features
from sklearn.ensemble import RandomForestClassifier
import config
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score

readmission_data = preprocess('../data/diabetic_data.csv')

groups = readmission_data['patient_nbr']
X = readmission_data.drop(columns= ['readmitted', 'patient_nbr'], axis= 1)
y= readmission_data['readmitted']

gss = GroupShuffleSplit(n_splits= 1, test_size= 0.3, random_state= 42)

train_id, test_id = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_id], X.iloc[test_id]
y_train, y_test = y.iloc[train_id], y.iloc[test_id]

ct = encode_features(config.ORDERED_CATS, config.ORDINAL_ORDER, config.UNORDERED_CATS)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

rf = RandomForestClassifier(n_estimators= 100, class_weight= 'balanced', random_state= 42, max_depth= 10)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred)}")
print(f"Predicted positives: {y_pred.sum()}")
print(f"Actual positives: {y_test.sum()}")