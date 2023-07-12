from glob import glob
import cv2
import skimage
import os
import numpy as np

class Ransac:
    def __init__(self, estimator_model):
        self.model_estimator = estimator_model

    def fit(self, data, num_sample, threshold):
        num_iterations = 150 #math.inf
        iterations_done = 0

        max_inlier_count = 0
        best_model = None
        left_img_points = None
        right_img_points = None

        total_data = data
        data_size = len(total_data)

        # Adaptively determining the number of iterations
        while num_iterations > iterations_done:
            
            sample_data = self.get_sample_data(total_data, data_size, num_sample) 
            
            estimated_model = self.model_estimator.fit(sample_data)

            # count the inliers within the threshold
            left_img_inliers, right_img_inliers, inlier_count = self.model_estimator.evaluate(total_data, threshold)

            # check for the best model 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model
                left_img_points = left_img_inliers
                right_img_points = right_img_inliers

            iterations_done = iterations_done + 1

            
        left_img_points = np.array(left_img_points)
        right_img_points = np.array(right_img_points)
        return best_model, left_img_points, right_img_points

    
    def get_sample_data(self, data, data_size, num_sample):
        generated_ind = set()
        itr = 0
        sample_data = []
        while itr < num_sample:
            ind = np.random.randint(data_size)
            if ind not in generated_ind:
                sample_data.append(data[ind])
                generated_ind.add(ind)
                itr += 1

        return np.array(sample_data)