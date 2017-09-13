import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

print("=== Get Raw Data")
datafiles = ["./raw/NSERC_GRT_FYR2016_AWARD.csv",
             "./raw/NSERC_GRT_FYR2015_AWARD.csv",
             "./raw/NSERC_GRT_FYR2014_AWARD.csv",
             "./raw/NSERC_GRT_FYR2013_AWARD.csv",
             "./raw/NSERC_GRT_FYR2012_AWARD.csv",
             "./raw/NSERC_GRT_FYR2011_AWARD.csv",
             "./raw/NSERC_GRT_FYR2010_AWARD.csv",
             "./raw/NSERC_GRT_FYR2009_AWARD.csv",
             "./raw/NSERC_GRT_FYR2008_AWARD.csv",
             "./raw/NSERC_GRT_FYR2007_AWARD.csv",
             ]
print("Using data files:")
dfs = []
for datafile in datafiles:
    print("- " + datafile)
    df = pd.read_csv(datafile, index_col=False, engine='python')
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print("Available Columns:")
for col_name in df.columns:
    print("* " + col_name)

print("=== Preprocess Data")
data_name = 'ApplicationSummary'
target_name = 'ResearchSubjectGroupEN'
df = df[[data_name, target_name]]
df = df.loc[df[data_name] != "No summary - Aucun sommaire"].loc[pd.notnull(df[data_name])]
print("Use %s as data and %s as target" % (data_name, target_name))
df = shuffle(df)
data = df[data_name]
target = df[target_name]
target_names = target
le = LabelEncoder()
target = le.fit_transform(target)
print("Target Classes:")
for name in le.classes_:
    print("* " + name)
test_split = 0.05
print("Set test split to %.2f" % (test_split))
num_test_cases = int(len(target)*test_split)
data_train, target_train = data[num_test_cases:], target[num_test_cases:]
data_test, target_test = data[:num_test_cases], target[:num_test_cases]
target_names_test = target_names[:num_test_cases]
print("Number of traning cases: %d" % (len(data_train)))
print("Number of test cases: %d" % (len(data_test)))

print("=== Training Classifier")
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, random_state=42,
                                          max_iter=10, tol=None)),
                    ])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
             }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1).fit(data_train, target_train)

print("=== Evaluate")
predicted = gs_clf.predict(data_test)
accuracy = np.mean(predicted == target_test)
print("Accuracy:", accuracy)
print(metrics.classification_report(target_test, predicted,
    target_names=le.classes_))
