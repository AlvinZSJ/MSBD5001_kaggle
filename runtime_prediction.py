import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse


def feature_encoding(df_train, feats_onehot):
    # One hot encoding
    for feature in feats_onehot:
        df_train = pd.concat([df_train, pd.get_dummies(df_train[feature], prefix=feature)], axis=1)

    return df_train


def feature_gen(df):
    """
    Feature generation

    :param df: df without feature generation
    :return: df with generated features
    """
    df.loc[df["n_jobs"] == -1, "n_jobs"] = 32
    df['clusters_job_q'] = df['n_clusters_per_class'] / df['n_jobs']
    df['iter_feature_m'] = df['max_iter'] * df['n_features']
    #
    df['sample_feature_classes_iter_formative_m_job_q'] = df['n_samples'] * df['n_features'] \
                                                        * df['n_informative'] * df['max_iter'] * df['n_classes'] / df['n_jobs']\
                                                        * (1 - df["penalty_l2"])

    df['iter_feature_samplel_m'] = df['max_iter'] * df['n_features'] \
                                    * df['n_samples'] * df['n_classes'] * (1 - (df['penalty_l2'] + df['penalty_none']))

    df["class_job_q"] = df["n_classes"] / df['n_jobs']
    df['iter_samples_m'] = df['max_iter'] * df['n_samples']
    df['class_l1_m'] = df['n_classes'] * df['penalty_l1']
    df['n_clusters'] = df['n_classes'] * df['n_clusters_per_class']
    df['alpha_feature_m'] = df['n_features'] * df["alpha"]
    df['sample_features_m'] = df["n_samples"] * df["n_classes"]
    df['alpha_iter_m'] = df['alpha'] * df['max_iter']

    df['n_useless'] = df['n_features'] / df['n_informative']
    df['n_flip_samples'] = df['flip_y'] * df['n_samples']

    return df


def process(df, cat_cols):

    df = feature_encoding(df, cat_cols)
    df = df.drop(columns=cat_cols)
    df = feature_gen(df)
    df = feature_scaling(df)
    return df


def feature_scaling(df_train, num_cols):
    sc = MinMaxScaler()
    for feature in num_cols:
        df_train[feature] = sc.fit_transform(np.array(df_train[feature]).reshape((-1, 1)))

    return df_train


def feature_scaling(df_train):
    sc = MinMaxScaler()
    for feature in list(df_train):
        df_train[feature] = sc.fit_transform(np.array(df_train[feature]).reshape((-1, 1)))

    return df_train


parser = argparse.ArgumentParser(description='Runtime Prediction')
parser.add_argument("train_csv", help="filename of training data")
parser.add_argument("test_csv", help="filename of test data")
args = parser.parse_args()

df_train = pd.read_csv(args.train_csv)
df_test = pd.read_csv(args.test_csv)

df_test_id = df_test['id']
df_test = df_test.drop(columns=['id'])

cat_cols = list(df_train.select_dtypes(include=['object']).columns)
num_cols = list(df_train.select_dtypes(exclude=['object']).columns)

df_time = df_train['time']
df_train = df_train.drop(columns=['time'])
df_train = process(df_train, cat_cols)
df_test = process(df_test, cat_cols)

df_train = df_train.drop(columns=['random_state', 'scale', 'l1_ratio',
                                        'flip_y', 'alpha', 'n_clusters_per_class'])

df_test = df_test.drop(columns=['random_state', 'scale', 'l1_ratio',
                                        'flip_y', 'alpha', 'n_clusters_per_class'])

X = df_train
Y = df_time


# define base model
def baseline_model(optimizer="RMSprop", init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(29, input_dim=23, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

# Nadam, RMSprop
seed = 7

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

shape = df_train.shape[1]
model = baseline_model()
model.summary()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
history = model.fit(df_train,
                    df_time,
                    epochs=1000,
                    batch_size=3,
                    validation_split=0.3,
                    verbose=1,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)


test_predictions = model.predict(df_test).flatten()

y_pred = model.predict(df_test)
df_pred = pd.DataFrame(y_pred)
df_pred = abs(df_pred)
df_pred = pd.concat([df_test_id, df_pred], axis=1)
df_pred.columns = ['Id', 'time']
df_pred.to_csv('Submission.csv', index=False)
