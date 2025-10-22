import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2

from parmap import parmap
########  MULTIPROCESSING ################
import multiprocessing
from multiprocessing import Pool, cpu_count
##########################################

import functools
import warnings
warnings.filterwarnings("ignore")

# passing all the arguments here
##Question:  Do we have any other method to do this.

def parseArgs(parser):
    parser.add_argument('--labels_test', default='icdar17_labels_test.txt',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', default='icdar17_labels_train.txt',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test', default='icdar17_local_features/test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train', default='icdar17_local_features/train',
                        help='the input folder of the training images / features')
    parser.add_argument('--bonus_train', default="icdar17_new/icdar2017-training-binary",
                        help='the input folder of the bonus training images')
    parser.add_argument('--bonus_test', default="ScriptNet-HistoricalWI-2017-binarized",
                        help='the input folder of the bonus testing images')
    parser.add_argument('--powernorm', default=True, action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    parser.add_argument('--overwrite', default=True, action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--evaluate', default=True,
                        help='Evaulate the encodings')
    parser.add_argument('--create_new_mus', default=True,
                        help='Creating new mus')
    return parser

def computeDescs(filename): #for single image
    """
    Compute SIFT descriptors for an image with specific keypoints and normalization.
    Parameters:
        filename (str): Path to the image file.
    Returns:
        np.ndarray: SIFT descriptors.
    """
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {filename}")

    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    for kp in keypoints:
        kp.angle = 0

    _, desc = sift.compute(image, keypoints)
    if desc is None:
        return None

    # Apply Hellinger normalization
    desc /= (np.sum(desc, axis=1, keepdims=True) + 1e-7)
    desc = np.sign(desc) * np.sqrt(np.abs(desc))

    return desc

def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """

    # print current directory
    print('current directory: {}'.format(os.getcwd()))
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors_bonus(files, max_descriptors):
    """
    Load random descriptors using the computeDescs function.
    Parameters:
        files (list): List of filenames containing images.
        max_descriptors (int): Maximum number of descriptors to load.
    Returns:
        np.ndarray: QxD matrix of descriptors.
    """
    max_files = 100
    indices = np.random.permutation(min(max_files, len(files)))
    files = np.array(files)[indices]
    print(files.shape)

    max_descs_per_file = int(max_descriptors / len(files))
    descriptors = []
    
    for i in tqdm(range(len(files))):
        print("======================================================")
        try:
            desc = computeDescs(files[i])
        except ValueError as e:
            print(e)
            continue

        if desc is not None:
            indices = np.random.choice(len(desc), min(len(desc), max_descs_per_file), replace=False)
            desc = desc[indices]
            descriptors.append(desc)
    print(len(descriptors))

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def vlad_bonus(files, mus, powernorm, gmp=False, gamma=1000):
    print(".............I am into the VLAD Bonus.............")    
    
    K = mus.shape[0]
    encodings = []
#descriptor ko dobara calculate kerna padega kyunki loadrandodesc me humne kuch hi loadkiye the jisse hum kuch centers calculate karenge and each descriptor ke around vo centers ko use karenge
    
    for f in tqdm(files):
        try:
            desc = computeDescs(f)
        except ValueError as e:
            print(e)
            continue

        if desc is None:
            continue

        a = assignments(desc, mus)
        T, D = desc.shape
        f_enc = np.zeros((D * K), dtype=np.float32)

        for k in range(mus.shape[0]):
            closest = desc[np.where(a[:, k] == 1)[0]]
            diff = closest - mus[k]
            #we have to compute the effect of GMP with and without exempler classifier
            if gmp:
                # Generalized Max Pooling with Ridge Regression
                ridge = Ridge(alpha=gamma, solver='sparse_cg', fit_intercept=False, max_iter=500)
                ridge.fit(diff, np.ones_like(diff))
                coef = ridge.coef_
                f_enc[k * D:(k + 1) * D] = coef.flatten()
            else:
                f_enc[k * D:(k + 1) * D] = np.sum(diff, axis=0)

        if powernorm:
            sign = np.where(f_enc >= 0, 1, -1)
            f_enc = sign * np.sqrt(np.abs(f_enc))

        f_enc /= np.linalg.norm(f_enc)
        encodings.append(f_enc)

    return encodings

#### random selection of local feature descriptors from a list of files ####
def loadRandomDescriptors(files, max_descriptors):
    """
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []

    # check all the files in current diretory
    # for file in os.listdir(os.getcwd()):
    #   print(file)

    for i in tqdm(range(len(files))):

        # check if fil exists
        if not os.path.exists(files[i]):
            print('file does not exist: {}'.format(files[i]))

        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')

        # get some random ones
        indices = np.random.choice(len(desc),min(len(desc),int(max_descs_per_file)),replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors #matrix of descriptors 1 rows = descriptor (Q) ||| 1 column = dimension (D) of that descriptor
# clusters: refers to group of similar data points
# we will try to find the clusters and then return the centers mus (KxD)
# center are the visual words
# the cluster centers are (100, 128)
def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    # use MiniBatchKMeans for faster clustering
    print('Here for clustering the descriptors')
    kmeas_descriptors = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
    kmeas_descriptors.fit(descriptors)
    # get the center of the clusters
    centers = kmeas_descriptors.cluster_centers_
    print('Descriptors shape:{} and cluster centers shape: {}'.format(descriptors.shape, centers.shape))  
    return centers  # mus are the centers  number of center is K

# this function effectively labels each descriptor with index of its closest clusters
def assignments(descriptors, clusters):  # here cluster reperesent the center of the cluster
    
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """

    BfMatcher = cv2.BFMatcher()  # brute force matcher (matching descriptors and clusters)

    # compute nearest neighbor KNN for each descriptor
    nearest = BfMatcher.knnMatch(descriptors, clusters, k=1)

    # create assignment matrix
    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.float32)

    # set 1 where the nearest neighbor
    for i, nn in enumerate(nearest):
        # trainIdx is the index of the cluster which is the nearest neighbor
        assignment[i, nn[0].trainIdx] = 1

    return assignment  # matrix indicated nearest cluster for each discriptor

'''#bag of visual words technique: clustering descriptors and finding cluster centers to create
compact representation of the dataset to reduce the complexity'''

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:  # rb is read binary
            desc = cPickle.load(ff, encoding='latin1')
        # compute the nearest cluster for each descriptor
        a = assignments(desc, mus)  ################assignments

        T, D = desc.shape

        f_enc = np.zeros((D * K), dtype=np.float32) #initialisation

        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select
            # TODO

            # select only those descriptors that have this cluster as nearest neighbor
            closest = desc[np.where(a[:, k] == 1)[0]]  # find the row indices in a where the elements in the kth column are 1
            # compute the difference to the cluster center (residuals)
            diff = closest - mus[k]

            if args.gmp:
                # generalized max pooling
                # TODO for individual
                pass
            else:
                # sum pooling
                #print('inside the aggregation (sum pooling)')
                f_enc[k * D:(k + 1) * D] = np.sum(diff, axis=0)  # sum along rows
        # c) power normalization
        if powernorm:
            #print('inside the power norm' + str(powernorm))
            sign = np.where(f_enc >= 0, 1, -1)
            f_enc = sign * np.sqrt(np.abs(f_enc))

        # l2 norm
        f_enc /= np.linalg.norm(f_enc)

        encodings.append(f_enc)
    return encodings

# computing the svm for a individual global representaion (encoding)
def train_svm(train_enc, test_enc, C_value):
    # we tried the (1,-1)
    n_train = len(train_enc)
    labels = np.ones(n_train + 1)
    labels[:-1] = -1  # train_Enc labels should be -1

    #if train_enc.shape[1] != test_enc.shape[0]:
        #raise ValueError(f"Feature dimension mismatch: train_enc has {train_enc.shape[1]} features, test_enc has {test_enc.shape[0]} features")

    stacked_train_data = np.vstack((train_enc, test_enc))
    svm = LinearSVC(C=C_value, class_weight='balanced')
    svm.fit(stacked_train_data, labels)
    # print(svm.coef_.shape)
    new_global_desc = normalize(svm.coef_, norm='l2')

    return new_global_desc.flatten()

# exemplar classification
def esvm(encs_test, encs_train, C=1000):
    # Set the number of jobs to the number of CPUs if n_jobs is -1

    n_cpu = os.cpu_count()
    print("Number of CPUs: ", n_cpu)

    with multiprocessing.Pool(processes=n_cpu) as pool:
        new_global_descriptors = pool.starmap(train_svm, [(encs_train, enc, C) for enc in encs_test])
    print("multiprocessing")

    return new_global_descriptors

def distances(encs):
    """
    compute pairwise distances

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized encodings
    # TODO done done done
    # mask out distance with itself
    dists = 1 - np.dot(normalize(encs), normalize(encs).T)
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    if args.evaluate == False:
        print('Evaluation flag is set to false!!!!!!!!!!!!!!')
        return

    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    print(len(encs), len(labels))
    
    # Sort each row of the distance matrix and get the sorted indices
    indices = np.argsort(dist_matrix, axis=1)

    #indices = dist_matrix.argsort()

    #initialisation
    n_encs = len(encs)
    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)  # fix random seed
    print('--'*50)
    print('Arguments')
    print('args: {}'.format(args))
    print('--'*50)

     #intialisation
    n_clusters = 32 #can be 100 but needs a lot of computation power to complete it.
    max_descriptors = 500000 #500000

################## TRAIN DATA LOADING #############

    print('Loading original images from train bonus directory') 
    bonus_images_train = [os.path.join(args.bonus_train, f) for f in os.listdir(args.bonus_train) if f.endswith('.png')]
    print(f"number of train images: {len(bonus_images_train)}")

    print('--'*50)

#################    LABELS LOADING    #################

    print("<<<<<<<<<<<<<<  Labels Loading  >>>>>>>>>>")
    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    print('train files size: {}'.format(len(files_train)))
    print('train labels size: {}'.format(len(labels_train)))
    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('test files size: {}'.format(len(files_test)))
    print('test label size: {}'.format(len(labels_test)))

##############  MAKING MUS FOR THE test AND train DATA  ############ 

    if not os.path.exists('mus.pkl.gz') or args.create_new_mus:
        # TODO 
        print('Going for loading random descriptors right')
        print('> compute dictionary')
        # TODO
        descriptors_bonus = loadRandomDescriptors_bonus(bonus_images_train, max_descriptors)
        print('> loaded {} descriptors:'.format(len(descriptors_bonus)))
        print("calculating new mus on n_clusters")
        mus_bonus = dictionary(descriptors_bonus, n_clusters)

        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus_bonus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus_bonus = cPickle.load(f)

####################   VLAD Encoding TEST   ######################

    print('--'*50)
    print('<<<<<< compute VLAD for test using BONUS images >>>>>>')
    
    print('Loading original images from test bonus directory')
    bonus_dir = "ScriptNet-HistoricalWI-2017-binarized"                   #argparse

    bonus_images = [os.path.join(args.bonus_test, f) for f in os.listdir(args.bonus_test) if f.endswith('.jpg')]
    
    print(f"number of test images: {len(bonus_images)}")
   
    print("loading rando descriptors on max descriptors 5000")
    descriptors_bonus = loadRandomDescriptors_bonus(bonus_images, max_descriptors)
    

    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        print("Calcuatng new bonus test encodings on bonus images")
        encoding_bonus_test = vlad_bonus(bonus_images, mus_bonus, powernorm = True, gmp=False, gamma=1000)
        enc_test = encoding_bonus_test
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # cross-evaluate test encodings
    print('-|' * 50)
    print('<<<<<<<<<   evaluate test encodings    >>>>>>>>>>')

    evaluate(enc_test, labels_test)
    print('-|'*50)

##################    VLAD for TRAIN    #################
    
    print('<<<<<<<<<    compute VLAD for train using BONUS images    >>>>>>>>>>')

    fname_train = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname_train) or args.overwrite:
        # TODO
        print("Calcuatng new bonus train encodings on bonus images")
        encoding_bonus = vlad_bonus(bonus_images_train, mus_bonus, powernorm = True, gmp=False, gamma=1000)
        enc_train = encoding_bonus
        with gzip.open(fname_train, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname_train, 'rb') as f:
            enc_train = cPickle.load(f)

    # cross-evaluate test encodings
    print('-|' * 50)
    print('> evaluate train encodings this time with bonus')
    evaluate(enc_train, labels_train)
    print('-|' * 50)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    print('Encodings shape: Test encodings {} and Train encodings {}'.format(len(enc_test), len(enc_train)))
    enc_test = esvm(enc_test, enc_train, args.C)

    print('-|' * 50)
    print('> evaluate results after ESVM')
    evaluate(enc_test, labels_test)

    print('-|' * 50)
