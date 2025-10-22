import numpy as np
import pickle

UNKNOWN_LABEL = -1  # Define a constant for unknown labels

class OpenSetEvaluation:

    def __init__(self, classifier, false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):
        self.false_alarm_rate_range = false_alarm_rate_range
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []
        self.classifier = classifier

    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='latin1')

    def run(self):

        similarity_thresholds = []
        identification_rates = []

        # Compute the similarity between the embeddings of the test set and the training set.
        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # Compute the identification rate for different similarity thresholds.

        for false_alarm_rate in self.false_alarm_rate_range:
            similarity_threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            threshold_lables = np.where(similarities >= similarity_threshold, prediction_labels, UNKNOWN_LABEL)
            identification_rate = self.calc_identification_rate(threshold_lables)

            identification_rates.append(identification_rate)
            similarity_thresholds.append(similarity_threshold)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results
    def select_similarity_threshold(self, similarity, false_alarm_rate):
        # Compute the threshold for the given false alarm rate.
        unknown_similarity = similarity[np.where(self.test_labels == UNKNOWN_LABEL)[0]]
        threshold = np.percentile(unknown_similarity, 100 * (1 - false_alarm_rate))
        return threshold

    def calc_identification_rate(self, prediction_labels):
        # compute identification rate using the prediction labels

        known_indices = np.where(self.test_labels != UNKNOWN_LABEL)[0]

        correct_predictions = sum(prediction_labels[i] == self.test_labels[i] and prediction_labels[i] != UNKNOWN_LABEL for i in known_indices)

        identification_rate = correct_predictions / len(known_indices)

        return identification_rate
