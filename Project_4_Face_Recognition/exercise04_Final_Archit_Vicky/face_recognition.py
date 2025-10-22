import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from scipy.spatial import distance

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    #num_neighbours = 50 can be submitted
    def __init__(self, num_neighbours=11, max_distance=0.6, min_prob=0.7, expected_label="Unknown"):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()

        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        # The underlying gallery: class labels and embeddings.

        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

        #additional to check accuracy
        self.correct = 0
        self.total = 0
        self.expected_label = expected_label
        print("Expected label: ", self.expected_label)

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        self.labels.append(label)
        self.embeddings = np.vstack((self.embeddings, self.facenet.predict(face)))
        return

    # ToDo
    def predict(self, face):

        #Use KNN to assign class label to aligned face
        #Hint: Use scipy.spatial.distance.cdist to calculate distance between embeddings

        embedding = self.facenet.predict(face)
        dist = distance.cdist(embedding.reshape(1, -1), self.embeddings, 'cosine')
        print(dist.shape)

        #knn
        # Get the indices of the top k nearest/closest neighbors
        indices = np.argsort(dist)
        indices = indices[0][:self.num_neighbours]
        print(indices)

        # Get the names of the top k nearest neighbors
        labels = [self.labels[i] for i in indices]
        print(labels)

        # Get the unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(unique_labels, counts)

        # Get the label with the maximum count
        predicted_label = unique_labels[np.argmax(counts)]
        print(predicted_label)

        # Get the probability of the predicted label
        prob = counts[np.argmax(counts)] / self.num_neighbours

        # Get the distance to the prediction
        dist_to_prediction = dist[0][indices[np.argmax(counts)]]



        # if distance is greater than threshold set to unknown as per 4.2 C
        # if probability is less than threshold set to unknown as per 4.2 C
        if np.min(dist_to_prediction) > self.max_distance or prob < self.min_prob:
            print('Either distance is large or prob is low so Unknown')
            predicted_label = "Unknown"

        #additional to check accuracy
        if predicted_label == self.expected_label:
            self.correct += 1

        #get accuracy
        try:
            self.total += 1
            accuracy = self.correct / self.total * 100
        except ZeroDivisionError:
            accuracy = 0
        print("Accuracy: ", accuracy)

        return predicted_label, prob, dist_to_prediction


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()  # normalised embeding

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:  # change: r to rb
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        # self.labels.append(label)
        self.embeddings = np.vstack((self.embeddings, self.facenet.predict(face)))
        return None

    print("Code chalrha h")

    # ToDo
    # need to debug the fit
    def fit(self):
        # calcualte the initial cluster centers, choose k embeddings randomly
        # store the estimated cluster center
        # labels assigned to the faces
        # randomly selects two indices from self.embeddings length
        random_indices = np.random.choice(self.embeddings.shape[0], self.num_clusters, replace=False)
        # initialization of cluster centers
        cluster_center = self.embeddings[random_indices]

        # initializing the array labels
        labels = np.zeros(self.embeddings.shape[0])

        # calculating the nearst cluster for each iterations and update the cluster center and labels
        try:
            for i in range(self.max_iter):
                for embedding in self.embeddings:
                    # calculate the nearest center and assign it to the
                    j = 0
                    distances = []
                    for center in cluster_center:
                        difference = np.linalg.norm(embedding - center)
                        distances.append(difference)
                        # assign the index of distance of the closest cluster center to labels[i]
                    labels[j] = np.argmin(distances)
                    j += 1

                    # update the cluster center
                    # itrate over the range num_clusters
                    for c in range(self.num_clusters):
                        # calculate the mean of all data point which were assigned the labels
                        data_point = self.embeddings[labels == c]
                        if len(data_point) == 0:
                            continue
                        new_centers = np.mean(data_point, axis=0)
                    # store the final centers and labels in attributes
                    #self.centers = new_centers
                    #self.labels = labels

                    self.cluster_center = new_centers
                    self.cluster_membership = labels
        except Exception as e:
            print(e)

                # k means working:
                # choose the k (number of cluster)
                # select initial cluster center
                # for each data point calculate the distance between datapoint and cluster center
                # assign each datapoint to the cluster whose center is closest to the data point
                # update the cluster center: mean of all data point assigned to each cluster
                # ....which becomes the new cluster center
                # repeat the steps 2 and 3
                # output = final clusters center and cluster assignment for each data points

    # ToDo
    def predict(self, face):
        embedding = self.facenet.predict(face)
        distances = []  # distribution of distances
        # going through all the centers to calculate the difference bw particular embedding and center

        for center in self.cluster_center:
            # calcualting the distances from a embedding to each center then looking which one is the nearest
            difference = np.linalg.norm(embedding - center)
            distances.append(difference)

        # cluster of interest is with the lowest distances
        cluster_of_interest = np.argmin(distances)
        return cluster_of_interest, distances

        # size of cluster center :
        # size of center: should be 128
        # embedding size = 128
        return None