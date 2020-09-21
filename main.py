import cv2
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import scale
from sklearn import preprocessing

from scipy.cluster.vq import vq
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle

# define some constants
CURRENT_PATH = os.getcwd()
IMG_PATH = os.path.join(CURRENT_PATH, 'data/images/')
IMG_LABELS_PATH = os.path.join(CURRENT_PATH, 'data/new_labels.csv')
IF_RANDOM_SPLIT = False
IF_RUN_CAL_ALL_DESCRIPTORS = False
IF_RUN_CAL_CODEBOOK = False
IF_TUNE_SVM = False
IF_TRAIN_SVM = True
IF_TUNE_KMEANS = False


def train_test_idx(all_img_idx, train_percent, if_random_split=IF_RANDOM_SPLIT):
    """

    :param all_img_idx: list of image index integers
    :param train_percent:
    :return:
    """
    if not if_random_split:
        np.random.seed(1)
    num_train = int(len(all_img_idx) * train_percent)
    train_idx = np.random.choice(all_img_idx, num_train, replace=False)
    train_idx.sort()
    test_idx = np.setdiff1d(all_img_idx, train_idx)
    print('train_idx: {}'.format(len(train_idx)))
    print('test_idx: {}'.format(len(test_idx)))
    return train_idx, test_idx


def cal_SIFT(img):
    """
    Get SIFT descriptors of a single image.
    :param img: image
    :return: descriptors of size m * 128, m is the number of keypoints of the image
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return des


def cal_SIFT_all(train_idx):
    print('\nCalculating SIFT for all images:')
    descriptors_list = []
    for idx in train_idx:
        if idx % 100 == 0:
            print(idx)
        try:
            img_path = IMG_PATH + 'img_{}.jpg'.format(str(idx).zfill(4))
            image = cv2.imread(img_path, 0)
            des = cal_SIFT(image)
            # if des is not None:
            descriptors_list.append(des)  # descriptors = np.vstack((descriptors, des))
            # this_img_idx = np.array([idx + 1] * des.shape[0]).reshape((-1, 1))
            # img_idx_list.append(this_img_idx)  # img_idx = np.vstack((img_idx, this_img_idx))
        except cv2.error as e:
            print('Image {} error! '.format(idx), e)
    descriptors = np.concatenate(descriptors_list, axis=0)
    print('descriptors.shape: {}'.format(descriptors.shape))
    print('Calculating SIFT for all images completed!')
    return descriptors


def tune_kmeans(all_descriptors):
    """
    Tune sklearn kmeans to get optimal cluster size, which is the codebook size
    :param all_descriptors:
    :return:
    """

    k_list = [5, 10, 20, 40, 60]
    sse = []
    for k in k_list:
        start_ts = datetime.datetime.now()
        print('\nRunning kmeans with cluster {}:'.format(k))
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(all_descriptors)
        sse.append(kmeans.inertia_)
        print('cluster {}: sse is {}'.format(k, sse))
        end_ts = datetime.datetime.now()
        print('time of running : {}'.format(end_ts - start_ts))
        np.save('./output/sse.npy', sse)
    plt.plot(k_list, sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.savefig('./output/tune_kmeans.png')


def cal_codebook(all_descriptors, kmeans_model, num_clusters=40):
    all_descriptors = all_descriptors.astype(np.float32)

    # Apply KMeans
    if kmeans_model == 'cv2':
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        print('\nApplying cv2.kmeans:')
        start = datetime.datetime.now()
        compactness, labels, centers = cv2.kmeans(all_descriptors, num_clusters, None, criteria, 10, flags)
        print(labels.shape)
        #print(labels)
        # print(centers)
        end = datetime.datetime.now()
        elapsed_time = end-start
        print('kmeans elapsed time is: {}'.format(elapsed_time))
        print('cv2.kmeans completed!')
        return centers


    elif kmeans_model == 'sklearn':
        print('\nApplying skleran.KMeans:')
        start = datetime.datetime.now()
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-2)
        kmeans.fit(all_descriptors)
        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)
        print('sse is {}'.format(kmeans.inertia_))
        end = datetime.datetime.now()
        print('time of running kmeans: {}'.format(end-start))
        print('skleran.KMeans completed!')
        return kmeans.cluster_centers_


def cal_img_features(img, codebook):
    """
    Calculate the features of a single image given the codebook(vocabulary) generated by clustering method (kmeans), each column is the center of the cluster.
    :param img:
    :param codebook:
    :return:
    """
    features = np.zeros((1, codebook.shape[1]))
    SIFT = cal_SIFT(img)
    code, _ = vq(SIFT, codebook)
    for i in code:
        features[0, i] += 1
    return(features)


def cal_img_features_all(img_idx, codebook):
    print('\nStart calculating all image features:')
    features_all_list = []
    for idx in img_idx:
        img_path = IMG_PATH + 'img_{}.jpg'.format(str(idx).zfill(4))
        image = cv2.imread(img_path, 0)
        this_features = cal_img_features(image, codebook)
        features_all_list.append(this_features)
    features_all = np.concatenate(features_all_list, axis=0)
    print('features all shape is: {}'.format(features_all.shape))
    print('Calculating all image features completed!')
    return features_all


def get_df_labels(img_idx):
    # df_labels = pd.read_csv(IMG_LABELS_PATH)
    # df_labels = np.array(df_labels.iloc[img_idx, 1]).reshape((-1,1))
    # return(df_labels)
    df_labels = pd.read_csv(IMG_LABELS_PATH)
    df_labels = df_labels[df_labels['Image'].isin(img_idx)]
    df_labels = np.array(df_labels.iloc[:, 1]).reshape((-1, 1))
    return df_labels





def get_processed_df(img_idx, codebook):
    """
    Get processed dataframe for modeling.
    :param img_idx:
    :param codebook:
    :return:
    """
    features_all = cal_img_features_all(img_idx, codebook)
    df_labels = get_df_labels(img_idx)

    print('df_labels.shape: {}'.format(df_labels.shape))
    print('features_all.shape: {}'.format(features_all.shape))
    df = np.concatenate((features_all, df_labels), axis=1)
    #print(df)
    return df

def train_svm(train_df, C=10, if_tune=IF_TUNE_SVM):
    print('\nStart training SVM:')
    if if_tune:
        print('Tuning SVM:')
        scaler = preprocessing.StandardScaler().fit(train_df[:,:-1])
        X_train = scaler.transform(train_df[:, :-1])
        # X = scale(train_df[:,:-1], axis=0)
        with open('./output/scaler', 'wb') as f:
            pickle.dump(scaler, f)
        Y_train = train_df[:, -1]
        SVM = svm.SVC(kernel='linear')
        svc_param_grid = {'C': [0.01, 0.1, 1, 10, 100, 200, 500]}
        gsSVM = GridSearchCV(SVM, param_grid=svc_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
        gsSVM.fit(X_train, Y_train)
        print('grid search best score is: {}'.format(gsSVM.best_score_))
        print('grid search best model is: {}'.format(gsSVM.best_estimator_))
        # SVM.fit(X, Y)
        print('Tuning SVM completed!')
        return gsSVM
    else:
        scaler = preprocessing.StandardScaler().fit(train_df[:, :-1])
        X_train = scaler.transform(train_df[:, :-1])
        # X = scale(train_df[:,:-1], axis=0)
        with open('./output/scaler', 'wb') as f:
            pickle.dump(scaler, f)
        Y_train = train_df[:, -1]
        SVM = svm.SVC(kernel='linear')
        SVM.fit(X_train, Y_train)
        return SVM
    print('Training SVM completed!')

def predict_svm(model, test_df, scaler):
    print('\nStart predicting:')
    SVM = model
    X_test = scaler.transform(test_df[:, :-1])
    Y_test = test_df[:, -1]
    Y_pred = SVM.predict(X_test)
    accuracy = sum(Y_pred == Y_test)/len(Y_test)
    print('accuracy is: {}'.format(accuracy))
    print('Predicting completed!')
    return accuracy










if __name__ == '__main__':
    all_img_idx = list(pd.read_csv(IMG_LABELS_PATH)['Image'])[:3000]
    train_idx, test_idx = train_test_idx(all_img_idx, 0.8)
    if IF_RUN_CAL_ALL_DESCRIPTORS:
        all_descriptors = cal_SIFT_all(train_idx)
        # with open('./output/all_descriptors', 'wb') as f:
        #     pickle.dump(all_descriptors, f)
    # else:
    #     with open('./output/all_descriptors', 'rb') as f:
    #         all_descriptors = pickle.load(f)

    if IF_RUN_CAL_CODEBOOK:
        codebook = cal_codebook(all_descriptors, 'sklearn', 40)
        with open('./output/codebook', 'wb') as f:
            pickle.dump(codebook, f)
    else:
        with open('./output/codebook', 'rb') as f:
            codebook = pickle.load(f)

    train_df = get_processed_df(train_idx, codebook)
    SVM = train_svm(train_df)
    test_df = get_processed_df(test_idx, codebook)
    with open('./output/scaler', 'rb') as f:
        scaler = pickle.load(f)
    predict_svm(SVM, test_df, scaler)

