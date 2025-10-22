import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument('--powernorm', default=True, action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', default=True, action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--evaluate', default=True,
                        help='Evaulate the encodings')
    # parser.add_argument('--create_new_mus', default=False,
    #                     help='Creating new mus')
    return parser


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


def vlad(files, mus, powernorm, gmp = False, gamma=1000):
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
                # Generalized Max Pooling with Ridge Regression
                ridge = Ridge(alpha=gamma, solver='sparse_cg', fit_intercept=False, max_iter=500)
                #ridge.fit(diff, np.ones_like(diff))
                if (np.where(a[:, k] == 1)[0]).size == 0:
                    #print(">>>>>>>>>>>>>>>>>>>Skipiiiiiiinggggggg>>>>>>>>>>>>>>>")
                    continue
                ridge.fit(diff, np.ones(len(np.where(a[:, k] == 1)[0])))
                coef = ridge.coef_
                f_enc[k * D:(k + 1) * D] = coef.flatten()
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

def five_codebooks(descriptors, n_clusters):
    """
    Generate five different codebooks using MiniBatchKMeans.
    Parameters:
        descriptors (np.ndarray): NxD matrix of local descriptors.
        n_clusters (int): Number of clusters.
    Returns:
        list: List of five KxD matrices of cluster centers.
    """
    codebooks = []
    for i in range(5):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=i)
        kmeans.fit(descriptors)
        codebooks.append(kmeans.cluster_centers_)
    return codebooks

def five_vlad(files, max_descriptors, n_clusters, powernorm, gmp, gamma=1000):
    """
    Compute five VLAD encodings using different codebooks and concatenate them.
    Parameters:
        files (list): List of N files containing each T local descriptors of dimension D.
        max_descriptors (int): Maximum number of descriptors to load for clustering.
        n_clusters (int): Number of clusters for k-means.
        powernorm (bool): Whether to use power normalization.
        gmp (bool): Whether to use generalized max pooling.
        gamma (float): Regularization parameter for GMP.
    Returns:
        np.ndarray: Concatenated VLAD encodings.
    """
    # Load random descriptors
    print("Loading descriptors")
    descriptors = loadRandomDescriptors(files, max_descriptors)
    print('> loaded {} descriptors:'.format(len(descriptors)))

    # Generate five different codebooks
    print("loading codebook")
    codebooks = five_codebooks(descriptors, n_clusters)

    # Compute VLAD encodings for each codebook
    all_encodings = []
    for codebook in codebooks:
        #print("vlad started")
        encodings = vlad(files, codebook, powernorm, gmp, gamma)
        all_encodings.append(encodings)

    # Concatenate the encodings along the 0-th axis
    concatenated_encodings = np.concatenate(all_encodings, axis=1)
    return concatenated_encodings

# computing the svm for a individual global representaion (encoding)
def train_svm(train_enc, test_enc, C_value):
    # we tried the (1,-1)
    n_train = len(train_enc)
    labels = np.ones(n_train + 1)
    labels[:-1] = -1  # train_Enc labels should be -1

    stacked_train_data = np.vstack((train_enc, test_enc))
    svm = LinearSVC(C=C_value, class_weight='balanced')
    svm.fit(stacked_train_data, labels)
    # print(svm.coef_.shape)
    new_global_desc = normalize(svm.coef_, norm='l2')

    return new_global_desc.flatten()

def esvm(encs_test, encs_train, C=1000):
    # Set the number of jobs to the number of CPUs if n_jobs is -1

    n_cpu = os.cpu_count()
    print("Number of CPUs: ", n_cpu)

    with multiprocessing.Pool(processes=10) as pool:
        new_global_descriptors = pool.starmap(train_svm, [(encs_train, enc, C) for enc in encs_test])
    print("multiprocessing")

    return new_global_descriptors

def apply_pca(train_encodings, test_encodings, n_components=1000):
    # Standardize the data
    scaler = StandardScaler()
    train_encodings = scaler.fit_transform(train_encodings)
    test_encodings = scaler.transform(test_encodings)

    # Apply PCA with whitening
    pca = PCA(n_components=n_components, whiten=True)
    train_encodings_pca = pca.fit_transform(train_encodings)
    test_encodings_pca = pca.transform(test_encodings)

    return train_encodings_pca, test_encodings_pca

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
    #print("we are in the evaluate function")
    #print(len(encs), len(labels))
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
    indices = dist_matrix.argsort()

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

    n_clusters = 32 #can be 100 but needs a lot of computation power to complete it.
    max_descriptors = 500000 #500000
    
    print('load data')
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,args.labels_train)
    print('train files size: {}'.format(len(files_train)))
    print('train labels size: {}'.format(len(labels_train)))
    print('--'*50)

####uncomment this only not for five_vlad

    # if not os.path.exists('mus.pkl.gz') or args.create_new_mus:
    #     # TODO Done Done Done
    #     # load descriptors
    #     print('Going for loading random descriptors')
    #     descriptors = loadRandomDescriptors(files_train, 5000)  # later change to 500000
    #     print('> loaded {} descriptors:'.format(len(descriptors)))

    #     # cluster centers
    #     print('> compute dictionary')
    #     # TODO Done Done Done
    #     # going for cluster centers = 10000
    #     k_means_cluster = 100
    #     # mus are the centers
    #     mus = dictionary(descriptors, k_means_cluster)  #########dictionary

    #     with gzip.open('mus.pkl.gz', 'wb') as fOut:
    #         cPickle.dump(mus, fOut, -1)
    # else:
    #     with gzip.open('mus.pkl.gz', 'rb') as f:
    #         mus = cPickle.load(f)

    # -----------------VLAD Encoding-------------------------

    # b) VLAD encoding
    #print('--'*50)
    print('> compute VLAD for test')
    # archit
    # encodings = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('test files size: {}'.format(len(files_test)))
    print('test label size: {}'.format(len(labels_test)))

    fname = 'enc_test_gmp_multi{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz' #_multi can be remove
    if not os.path.exists(fname) or args.overwrite:
        # TODO

        encodings_test = five_vlad(files_test, max_descriptors, n_clusters, args.powernorm, args.gmp, args.gamma)
        #encodings = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
        enc_test = encodings_test
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # cross-evaluate test encodings
    print('-|' * 50)
    print('> evaluate test encodings')

    evaluate(enc_test, labels_test)
    #print('-|'*50)
    # _____________________Vlad for train_______________________
    print('> compute VLAD for train')
    # archit
    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    print('train files size: {}'.format(len(files_train)))
    print('train labels size: {}'.format(len(labels_train)))

    fname_train = 'enc_train_gmp_multi{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz' #_multi can be remove
    if not os.path.exists(fname_train) or args.overwrite:
        # TODO
        encodings_train = five_vlad(files_train, max_descriptors, n_clusters, args.powernorm, args.gmp, args.gamma)
        #encodings_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)
        enc_train = encodings_train
        with gzip.open(fname_train, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname_train, 'rb') as f:
            enc_train = cPickle.load(f)

    # cross-evaluate test encodings
    print('-|' * 50)
    print('> evaluate train encodings')
    evaluate(enc_train, labels_train)
    print('-|' * 50)
    print("<<<<<<<<<<<<<   CALCULATNG ESVM   >>>>>>>>>>>>>>>>>>")
    
    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    print('Encodings shape: Test encodings: {} and Train encodings: {}'.format(len(enc_test), len(enc_train)))
    print("<<<<<<<<<<<<<   PCA RUNNING   >>>>>>>>>>>>>>>>>>")
    enc_train, enc_test = apply_pca(enc_train, enc_test)
    enc_test = esvm(enc_test, enc_train, args.C)

    print('-|' * 50)
    #args.evaluate = True # only for testing
    print('> evaluate results after ESVM')
    evaluate(enc_test, labels_test)
    print('*|*' * 50)
