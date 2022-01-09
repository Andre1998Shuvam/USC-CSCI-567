import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels)

    fp = np.sum(np.array(real_labels == 0) & np.array(predicted_labels == 1))
    tp = np.sum(np.array(real_labels == 1) & np.array(predicted_labels == 1))
    fn = np.sum(np.array(real_labels == 1) & np.array(predicted_labels == 0))

    if tp == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return f1
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = np.linalg.norm(np.array(point1) - np.array(point2), 3)
        return d
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = np.linalg.norm(np.array(point1) - np.array(point2))
        return d
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if np.linalg.norm(point1) == 0 or np.linalg.norm(point2) == 0:
            d = 1

        else:
            d = 1 - np.dot(np.array(point1), np.array(point2)) / \
                (np.linalg.norm(np.array(point1))
                 * np.linalg.norm(np.array(point2)))
        return d
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
                (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        best_f1 = -1
        best_k = 1
        best_model = None
        best_dist = None

        for k in range(1, 30, 2):
            for d in range(len(list(distance_funcs.keys()))):
                model = KNN(k, list(distance_funcs.values())[d])
                model.train(x_train, y_train)
                y_val_pred = model.predict(x_val)
                f1 = f1_score(y_val, y_val_pred)

                if(f1 > best_f1):
                    best_f1 = f1
                    best_model = model
                    best_k = k
                    best_dist = list(distance_funcs.keys())[d]

                elif f1 == best_f1:
                    if d < list(distance_funcs.keys()).index(best_dist):
                        best_dist = list(distance_funcs.keys())[d]
                        best_model = model
                        best_k = k

                    elif list(distance_funcs.keys())[d] == best_dist:
                        best_k = best_k
                        best_model = best_model
                        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_dist
        self.best_model = best_model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        best_f1 = -1
        best_k = 1
        best_s = None
        best_model = None
        best_dist = None
        
        for s in scaling_classes:
            for k in range(1, 30, 2):
                for d in range(len(list(distance_funcs.keys()))):

                        scaler = scaling_classes[s]()
                        x_train_scaled = scaler(x_train)
                        x_val_scaled = scaler(x_val)
                        model = KNN(k, list(distance_funcs.values())[d])
                        model.train(x_train_scaled, y_train)
                        y_val_pred = model.predict(x_val_scaled)
                        f1 = f1_score(y_val, y_val_pred)

                        if(f1 > best_f1):
                            best_f1 = f1
                            best_model = model
                            best_k = k
                            best_s = s
                            best_dist = list(distance_funcs.keys())[d]

                        elif f1 == best_f1:

                            if(list(scaling_classes.keys()).index(s) < list(scaling_classes.keys()).index(best_s)):
                                best_model = model
                                best_k = k
                                best_s = s
                                best_dist = list(distance_funcs.keys())[d]

                            elif(s == best_s):
                                if d < list(distance_funcs.keys()).index(best_dist):
                                    best_dist = list(distance_funcs.keys())[d]
                                    best_model = model
                                    best_k = k

                                elif list(distance_funcs.keys())[d] == best_dist:
                                    best_k = min(best_k,k)
                                    if best_k == k:
                                        best_model = model
                                        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_dist
        self.best_scaler = best_s
        self.best_model = best_model

        # raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features, dtype='float').T
        norms = (np.linalg.norm(features, axis=0)) != 0
        features[:, norms] = features[:, norms] / \
            np.linalg.norm(features, axis=0)[norms]

        return list(features.T)
        raise NotImplementedError


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
                For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
                This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
                The minimum value of this feature is thus min=-1, while the maximum value is max=2.
                So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
                leading to 1, 0, and 0.333333.
                If max happens to be same as min, set all new values to be zero for this feature.
                (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        f = np.array(features, dtype='float')
        non_zero = np.max(f, axis=0) - np.min(f, axis=0) != 0
        zero = np.max(f, axis=0) - np.min(f, axis=0) == 0

        f[:, non_zero] = (f[:, non_zero] - np.min(f, axis=0)[non_zero]) / \
            (np.max(f, axis=0) - np.min(f, axis=0))[non_zero]
        f[:, zero] = 0

        return list(f)

        raise NotImplementedError
