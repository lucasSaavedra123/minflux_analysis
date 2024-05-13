import keras
from keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json
import pandas as pd
import numpy as np
import tqdm
from utils import transform_traj_into_features
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from sklearn.model_selection import train_test_split
import os
from Trajectory import Trajectory
from collections import defaultdict
import pickle
import glob


APPEND_MORE_DATA = False
CREATE_DATA = False
LOAD_MODEL = True
PLOT_STATS = True

DATASET_PATH = './single_simulations'
RAW_DATA_PATH = './dataset.npy'

CLASS_LABELS = ['HD', 'TD']

if CREATE_DATA:
    if not os.path.exists(DATASET_PATH):
        print(f"{DATASET_PATH} is not a valid path")
        exit()
    else:
        raw_arrays = []

        for _, single_dataset_path in tqdm.tqdm(list(enumerate(glob.glob(DATASET_PATH+'/*')))):
            table = pd.read_csv(single_dataset_path)
            raw_arrays.append(np.zeros((1, (60*2)+len(CLASS_LABELS))))

            trajectory_df = table.sort_values('t')
            new_array = np.zeros((1, len(trajectory_df), 3))
            new_array[0,:,0] = trajectory_df['x']
            new_array[0,:,1] = trajectory_df['y']
            new_array[0,:,2] = trajectory_df['t']

            #plt.plot(trajectory_df['x'], trajectory_df['y'])
            #plt.title(trajectory_df['label'].values[0].upper())
            #plt.show()

            try:
                raw_arrays[-1][0, :-len(CLASS_LABELS)] = transform_traj_into_features(new_array)[0]
            except AssertionError:
                raw_arrays[-1][0, :-len(CLASS_LABELS)] = np.nan

            label = trajectory_df['label'].values[0].upper()

            try:
                raw_arrays[-1][0, -len(CLASS_LABELS):] = np.eye(len(CLASS_LABELS), dtype=int)[CLASS_LABELS.index(label)]
            except IndexError:
                raise Exception(f'{trajectory_df["label"].values[0]} is wrong')

        RAW_ARRAY = np.concatenate(raw_arrays)

        if APPEND_MORE_DATA:
            if os.path.exists(RAW_DATA_PATH):
                OLD_DATA = np.load(RAW_DATA_PATH)
                RAW_ARRAY = np.append(OLD_DATA, RAW_ARRAY, axis=0)
            else:
                print("WARNING: Data is not appended because dataset.npy it does not exist")
        np.random.shuffle(RAW_ARRAY)
        np.save(RAW_DATA_PATH, RAW_ARRAY)
else:
    if os.path.exists(RAW_DATA_PATH):
        RAW_ARRAY = np.load(RAW_DATA_PATH)
    else:
        print("dataset.npy does not exist")
        exit()

X = RAW_ARRAY[~np.any(np.isnan(RAW_ARRAY), axis=1), :-len(CLASS_LABELS)]
Y = RAW_ARRAY[~np.any(np.isnan(RAW_ARRAY), axis=1), -len(CLASS_LABELS):]

print("Data shape", X.shape, Y.shape)
model = keras.Sequential(
    [   
        keras.Input(shape=(X.shape[1],)),
        layers.Dense(20, activation="sigmoid"),
        layers.Dense(20, activation="sigmoid"),
        layers.Dense(len(CLASS_LABELS), activation="softmax"),
    ]
)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

if LOAD_MODEL:
    try:
        model.load_weights('model.keras')
        X_val, Y_val = np.load('X_val_do_not_delete.npy'), np.load('Y_val_do_not_delete.npy')
        history_dict = json.load(open('training_history_do_not_delete.json', 'r'))
    except FileNotFoundError:
        print("LOAD_MODEL=True cannot be executed because files are measing. Please, set LOAD_MODEL=False")
else:
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20)
    history_dict = model.fit(X_train, Y_train, validation_data=[X_val, Y_val], epochs=5, batch_size=8).history
    json.dump(history_dict, open('training_history_do_not_delete.json', 'w'))
    model.save('model.keras')
    np.save('X_val_do_not_delete.npy', X_val)
    np.save('Y_val_do_not_delete.npy', Y_val)

if PLOT_STATS:
    number_of_epochs = len(history_dict['categorical_accuracy'])
    epochs_list = list(range(1,number_of_epochs+1))
    plt.plot(epochs_list, history_dict['categorical_accuracy'])
    plt.plot(epochs_list, history_dict['val_categorical_accuracy'])
    plt.title('Simplified CONDOR (Gentili, 2021) model accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.xlim([0,epochs_list[-1]+1])
    #plt.ylim([0,1])

    plt.show()

    ground_truth = np.argmax(Y_val, axis=-1)
    predicted = np.argmax(model.predict(X_val), axis=-1)

    confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

    normalized = True
    confusion_mat = np.round(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2) if normalized else confusion_mat

    confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=CLASS_LABELS, columns=CLASS_LABELS)
    sns.set(font_scale=1.5)
    color_map = sns.color_palette(palette="Blues", n_colors=7)
    sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

    plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predicted, average="micro"),2)})')
    plt.rcParams.update({'font.size': 15})
    plt.ylabel("Ground truth", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

results = []

def new_dict():
    return {label: 0 for label in CLASS_LABELS}

counter = defaultdict(new_dict)
traces, labels = [], []

p = {
    '_id': 1,
    'x':1,
    'y':1,
    't': 1,
    'info.dataset': 1,
    'info.classified_experimental_condition': 1,
    'info.analysis.betha': 1,
    'info.analysis.confinement-classification': 1,
}

for t_info in tqdm.tqdm(Trajectory._get_collection().find({'info.immobile':False}, p)):
    #t = Trajectory.objects(id=t_info['_id'])[0]
    if 'analysis' not in t_info['info'] or len(t_info['x']) <= 1 or 'confinement-classification' in t_info['info']['analysis']:
        continue

    confinement_classifications = []

    fake_trajectory = Trajectory(
        x=t_info['x'],
        y=t_info['y'],
        t=t_info['t'],
        noisy=True
    )

    for sub_t in fake_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]:
        raw_input = np.zeros((1, (60*2)))

        raw_sub_t = np.zeros((1, sub_t.length, 3))
        raw_sub_t[0,:,0] = np.array(sub_t.get_noisy_x()) * 1000
        raw_sub_t[0,:,1] = np.array(sub_t.get_noisy_y()) * 1000
        raw_sub_t[0,:,2] = np.array(sub_t.get_time())

        try:
            raw_input[0, :] = transform_traj_into_features(raw_sub_t)[0]
        except:
            confinement_classifications.append(None)
            continue

        predicted = np.argmax(model.predict(raw_input, verbose=0), axis=-1)
        confinement_classifications.append(CLASS_LABELS[predicted[0]])

    Trajectory._get_collection().update_one({'_id':t_info['_id']}, {"$set":{"info.analysis.confinement-classification":confinement_classifications}})

DatabaseHandler.disconnect()