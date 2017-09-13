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
from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical

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
data = [one_hot(text, 10000) for text in df[data_name]]
target = df[target_name]
target_names = target
le = LabelEncoder()
target = le.fit_transform(target)
print("Target classes:")
for name in le.classes_:
    print("* " + name)
num_classes = np.max(target) + 1
print("Number of classes:", num_classes)
test_split = 0.05
print("Set test split to %.2f" % (test_split))
num_test_cases = int(len(target)*test_split)
data_train, target_train = data[num_test_cases:], target[num_test_cases:]
data_test, target_test = data[:num_test_cases], target[:num_test_cases]
target_names_test = target_names[:num_test_cases]
print("Number of traning cases: %d" % (len(data_train)))
print("Number of test cases: %d" % (len(data_test)))

max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
data_train = tokenizer.sequences_to_matrix(data_train, mode='binary')
data_test = tokenizer.sequences_to_matrix(data_test, mode='binary')
print("data_train shape:", data_train.shape)
print("data_test shape:", data_test.shape)
target_train = to_categorical(target_train, num_classes)
target_test = to_categorical(target_test, num_classes)
print("target_train shape:", target_train.shape)
print("target_test shape:", target_test.shape)

print("=== Building model")
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(data_train, target_train,
                    batch_size=32,
                    epochs=100,
                    verbose=1,
                    validation_split=0.1)
print()
print("== Evalaute")
score = model.evaluate(data_test, target_test,
                       batch_size=32,
                       verbose=1)
print("== Evalaute")
print("Test score:", score[0])
print("Test accuracy:", score[1])
