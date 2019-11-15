# CSCI 3302: Homework 3 -- Clustering and Classification
# Implementations of K-Means clustering and K-Nearest Neighbor classification





import pickle
import random
import math
import numpy
import copy
import pdb
import matplotlib.pyplot as plt
from hw3_data import *
from queue import PriorityQueue

# TODO: INSERT YOUR NAME HERE
LAST_NAME = "mylastname"


def EuclideanDistance(p1, p2):
    return numpy.linalg.norm([p2[0]-p1[0], p2[1]-p1[1]])

def EuclideanDistance2(p1, p2):
    return numpy.linalg.norm(p1, p2)

def FindColMinMax(items):
    n = len(items[0]);
    minima = [sys.maxint for i in range(n)];
    maxima = [-sys.maxint - 1 for i in range(n)];

    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f];

            if (item[f] > maxima[f]):
                maxima[f] = item[f];

    return minima, maxima;


def UpdateMean(n, mean, item):
    for i in range(len(mean)):
        m = mean[i];
        m = (m * (n - 1) + item[i]) / float(n);
        mean[i] = round(m, 3);

    return mean;

def visualize_data(data, cluster_centers_file):
    fig = plt.figure(1, figsize=(4, 3))
    f = open(cluster_centers_file, 'rb')
    centers = pickle.load(f)
    f.close()

    km = KMeansClassifier()
    km._cluster_centers = centers

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    labels = []
    center_colors = []
    for pt in data:
        labels.append(colors[km.classify(pt) % len(colors)])

    for i in range(len(centers)):
        center_colors.append(colors[i])

    plt.scatter([d[0] for d in data], [d[1] for d in data], c=labels, marker='x')
    plt.scatter([c[0] for c in centers], [c[1] for c in centers], c=center_colors, marker='D')
    plt.title("K-Means Visualization")
    plt.show()


class KMeansClassifier(object):

    def __init__(self):
        self._cluster_centers = []  # List of cluster centers, each of which is a point. ex: [ [10,10], [2,1], [0,-3] ]
        self._data = []  # List of datapoints (list of immutable lists, ex:  [ (0,0), (1.,5.), (2., 3.) ] )
        self._centroid_dict = {}
        self.stopping_distance = .01

    def add_datapoint(self, datapoint):
        self._data.append(datapoint)

    def fit(self, k):
        # Fit k clusters to the data, by starting with k randomly selected cluster centers.
        self._cluster_centers = []  # Reset cluster centers array

        # First we initialize k points, called means, randomly.
        # We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far.
        # We repeat the process for a given number of iterations and at the end, we have our clusters.


        # TODO: Initialize k cluster centers at random points
        # HINT: To choose reasonable initial cluster centers, you can set them to be in the same spot as random (different) points from the dataset

        for i in range(0,k):
            random_centroid = self._data[random.randrange(0, len(self._data))]
            self._cluster_centers.append(random_centroid) # this appends for hte amont of k for how many random centroid we have based on k.
            self._centroid_dict[tuple(random_centroid)] = []



        # TODO Follow convergence procedure to find final locations for each center
        while True:
            ##########
            # step 1 #
            ##########
            # TODO: Iterate through each datapoint in self._data and figure out which cluster it belongs to
            # HINT: Use self.classify(p) for each datapoint p
            for point in self._data:
                best_centroid_idx = self.classify(point)
                self._centroid_dict[tuple(self._cluster_centers[best_centroid_idx])].append(point)

            ##########
            # step 2 #
            ##########
            # TODO: Figure out new positions for each cluster center (should be the average position of all its points)
            # this is euclidian distance between the two points
            new_centroids = []
            new_centroid_dict = {}
            change_in_centroids = []
            for old_centroid, points in self._centroid_dict.items():
                # calculate new centroid from the average of points
                avg_x = numpy.average([p[0] for p in points])
                avg_y = numpy.average([p[1] for p in points])
                new_centroid = [avg_x, avg_y]
                new_centroids.append(new_centroid)
                # set the new centroid as a key of the new_centroid dict
                new_centroid_dict[tuple(new_centroid)] = []
                change_in_centroids.append(EuclideanDistance(old_centroid, new_centroid))
            # TODO: Check to see how much the cluster centers have moved (for the stopping condition)
            meets_critera = []
            for change in change_in_centroids:
                if change < self.stopping_distance:
                    meets_critera.append(True)
                else:
                    meets_critera.append(False)
            stop = all(meets_critera)
            if stop:  # TODO: If the centers have moved less than some predefined threshold (you choose!) then exit the loop
                self._cluster_centers = new_centroids
                self._centroid_dict = new_centroid_dict
                break
            # set the new_centroid_dict as our self._centroid_dict
            self._cluster_centers = new_centroids
            self._centroid_dict = new_centroid_dict

    def classify(self, p):
        # Given a data point p, figure out which cluster it belongs to and return that cluster's ID (its index in self._cluster_centers)
        closest_cluster_index = 0
        curr_dist = 10000000
        best_centroid = None
        for centroid in self._cluster_centers:
            dist = EuclideanDistance(p, centroid)
            if dist < curr_dist:
                curr_dist = dist
                best_centroid = centroid
        return self._cluster_centers.index(best_centroid)
         # appends to the list that is the value of the random centroid key

        # TODO Find nearest cluster center, then return its index in self._cluster_centers


class KNNClassifier(object):

    def __init__(self):
        self._data = []  # list of (datapoint, label) tuples

    def clear_data(self):
        # Removes all data stored within the model
        self._data = []

    def add_labeled_datapoint(self, data_point, label):
        # Adds a labeled datapoint tuple onto the object's _data member
        self._data.append((data_point, label))

    def classify_datapoint(self, data_point, k):
        label_counts = {}  # Dictionary mapping "label" => vote count
        best_label = None

        # Perform k_nearest_neighbor classification, setting best_label to the majority-vote label for k-nearest points
        # TODO: Find the k nearest points in self._data to data_point


        pt_dist = []#PriorityQueue(maxsize=k) # instead of just an array a priority queue would be better?
        for pt_idx in range(len(self._data)): # picking class
            #for pt_idx in range(len(self._data[class_idx])): # iterating through points in class
                #pt = self._data[class_idx][pt_idx] # grabbing
                pt = self._data[pt_idx][0]
                pt_dist.append([pt_idx, EuclideanDistance(pt,data_point)])

        # TODO: Populate label_counts with the number of votes each label got from the k nearest points
        sorted_list = sorted(pt_dist, key=lambda x: x[1])
        k_nearest = sorted_list[0:k] # select the 0 to k nearest
        vote_dict = {}  # this would hold 'green' , 3 votes... for example.
        # TODO: Make sure to scale the weight of the vote each point gets by how far away it is from data_point
        #      Since you're just taking the max at the end of the algorithm, these do not need to be normalized in any way
        for idx in k_nearest:
            vote = self._data[idx[0]][1] #label
            weight = 1 * (1 / EuclideanDistance(pt, data_point)) #weight
            #if it already exists then we just add a weight count to that value.
            if(vote not in vote_dict):
                vote_dict[vote] = weight
            #if it is unique then add it to our dictionary
            else:
                vote_dict[vote] += weight
            best_label = min(vote_dict)
        #print("This is the weight2" + str(weight2))
        return best_label


def print_and_save_cluster_centers(classifier, filename):
    for idx, center in enumerate(classifier._cluster_centers):
        print("  Cluster %d, center at: %s" % (idx, str(center)))

    f = open(filename, 'wb')
    pickle.dump(classifier._cluster_centers, f)
    f.close()


def read_data_file(filename):
    f = open(filename)
    data_dict = pickle.load(f)
    f.close()

    return data_dict['data'], data_dict['labels']


def read_hw_data():
    global hw_data
    data_dict = pickle.loads(hw_data)
    return data_dict['data'], data_dict['labels']


def main():
    global LAST_NAME
    # read data file
    # data, labels = read_data_file('hw3_data.pkl')

    # load dataset
    data, labels = read_hw_data()

    # data is an 'N' x 'M' matrix, where N=number of examples and M=number of dimensions per example
    # data[0] retrieves the 0th example, a list with 'M' elements, one for each dimension (xy-points would have M=2)
    # labels is an 'N'-element list, where labels[0] is the label for the datapoint at data[0]

    ########## PART 1 ############
    # perform K-means clustering
    kMeans_classifier = KMeansClassifier()
    for datapoint in data:
        kMeans_classifier.add_datapoint(datapoint)  # add data to the model
    print(data[1])
    print(data[2])
    print(data[3])
    print(data[4])



    kMeans_classifier.fit(4)  # Fit 4 clusters to the data

    # plot results
    print('\n' * 2)
    print("K-means Classifier Test")
    print('-' * 40)
    print("Cluster center locations:")
    print_and_save_cluster_centers(kMeans_classifier, "hw3_kmeans_" + LAST_NAME + ".pkl")

    print('\n' * 2)

    ########## PART 2 ############
    print("K-Nearest Neighbor Classifier Test")
    print('-' * 40)

    # Create and test K-nearest neighbor classifier
    kNN_classifier = KNNClassifier()
    k = 2

    correct_classifications = 0
    # Perform leave-one-out cross validation (LOOCV) to evaluate KNN performance
    for holdout_idx in range(len(data)):
        # Reset classifier
        kNN_classifier.clear_data()

        for idx in range(len(data)):
            if idx == holdout_idx: continue  # Skip held-out data point being classified

            # Add (data point, label) tuples to KNNClassifier
            kNN_classifier.add_labeled_datapoint(data[idx], labels[idx])

        guess = kNN_classifier.classify_datapoint(data[holdout_idx], k)  # Perform kNN classification
        if guess == labels[holdout_idx]:
            correct_classifications += 1.0

    print("kNN classifier for k=%d" % k)
    print("Accuracy: %g" % (correct_classifications / len(data)))
    print('\n' * 2)

    visualize_data(data, 'hw3_kmeans_' + LAST_NAME + '.pkl')


if __name__ == '__main__':
    main()
