
from glob import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time, os

# reproducible results
rng = np.random.default_rng(42)

class FundamentalMatrix:
    
    def fit(self, data):
        self.F_mat = self.find_F_mat(data[:, :3], data[:, 3:])
        return self.F_mat
    
    def remove_outliers(self, data, threshold):
        """
        uses epipolar geometry and sampson distance to remove outliers and returns only inliers
        """
        x1 = np.array(data[:, :3])
        x2 = np.array(data[:, 3:])
        F = self.F_mat
        best_curr = []
        best_next = []
        num_inliers = 0
        
        for i in range(len(data)):
            # see Zisserman Multiple View Geometry CV 2nd Ed. pg. 287 (referenced by Szeliski)
            # use first order approximation of geometric error -- Sampson distance
            # distance between a point's epipolar line and its corresponding point
            # error = (x'.T * F * x)^2 / [(F * x_1)^2 + (F * x_2)^2 + (F.T * x'_1)^2 + (F.T * x'_2)^2]
            top = (x2[i].T @ F @ x1[i])**2
            # epipolar line 1
            F_x1 = F @ x1[i]
            # epipolar line 1
            F_x2 = F.T @ x2[i]
            bottom = F_x1[0]**2 + F_x1[1]**2 + F_x2[0]**2 + F_x2[1]**2
            error = top / bottom
            
            if error < threshold:
                best_curr.append(x1[i])
                best_next.append(x2[i])
                num_inliers += 1
        
        # input can be "array-like" and is returned as an array
        return np.ascontiguousarray(best_curr), np.ascontiguousarray(best_next), num_inliers

    def find_F_mat(self, kpoints1, kpoints2):
        """
        returns the normalized fundamental matrix calculated from keypoints 
        """
        mean1 = np.mean(kpoints1, axis=0)
        mean2 = np.mean(kpoints2, axis=0)

        # center pts at origin and scale so the mean distance between the origin and the points is 2 pixels
        mean_squared1 = np.mean(np.sum(kpoints1[:, :2], axis=1)) 
        scale1 = np.sqrt(2 / mean_squared1)
        
        mean_squared2 = np.mean(np.sum(kpoints2[:, :2], axis=1)) 
        scale2 = np.sqrt(2 / mean_squared2)        

        transform1 = np.array([[scale1, 0, -scale1*mean1[0]], [0, scale1, -scale1*mean1[1]], [0, 0, 1]])
        transform2 = np.array([[scale2, 0, -scale2*mean2[0]], [0, scale2, -scale2*mean2[1]], [0, 0, 1]])

        kpoints1 = (transform1 @ kpoints1.T).T
        kpoints2 = (transform2 @ kpoints2.T).T
        
        # x = point_list1[:, 0] 
        # y = point_list1[:, 1] 
        # x_p = point_list1[:, 0] # x'
        # y_p = point_list1[:, 1] # y'
        
        A = []
        for i in range(8):
            # (x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1)
            # A.append([x_p[i] * x[i], x_p[i] * y[i], x_p[i], y_p[i] * x[i], y_p[i] * y[i], y_p[i], x[i], y[i], 1])
            # matrix direct product is more accurate -- np.kron
            A.append(np.kron(kpoints2[i], kpoints1[i]))
        
        # SVD - find right singular vector corresponding to the smallest singular value
        U, S, V = np.linalg.svd(A)
        F = V.T[:, -1]
        
        F = np.reshape(F, (3,3))
        
        # take SVD of initial estimate 
        U, S, V = np.linalg.svd(F)
        S_diag = np.diag(S)
        
        # enforce the rank 2 constraint by making last singular value 0
        S_diag[2,2] = 0

        # F = U * E' * V
        F = U @ S_diag @ V
        
        # rescale
        F = transform2.T @ F @ transform1

        return F
    
    
class Ransac:
    
    def __init__(self, F_init):
        self.F = F_init

    def fit(self, data, thresh):
        """
        implement ransac for best fundamental matrix 
        """
        num_iterations = 150 
        max_inliers = 0
        points_curr = []
        points_next = []
        F_best = []

        for _ in range(num_iterations):
            
            # get 8 random matches
            inds = set()
            num_inds = 0
            random_sample = []
            while num_inds < 8:
                rints = rng.integers(len(data), size=1)
                ind = rints.item()
                if ind not in inds:
                    random_sample.append(data[ind])
                    inds.add(ind)
                    num_inds += 1
            
            random_sample = np.array(random_sample)
            
            F_first = self.F.fit(random_sample)

            # remove outliers and count inliers
            curr_inliers, next_inliers, num_inliers = self.F.remove_outliers(data, thresh)

            # evaluate 
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                F_best = F_first
                points_curr = curr_inliers
                points_next = next_inliers

        points_curr = np.array(points_curr)
        points_next = np.array(points_next)
        
        return F_best, points_curr, points_next   
    

class OdometryClass:
    
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
            
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)
    
    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        """
        extract ground truth coordinates from txt file
        """
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])  

    def get_scale(self, frame_id):
        """
        Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        """
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)
    
    def essential_matrix(self, k, F):
        """
        calculate and return essential matrix 
        """        
        E = k.T @ F @ k
        U, S, V = np.linalg.svd(E)
        S = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        E = U @ S @ V
        
        return E
               
    def camera_position(self, E):
        """
        use the essential matrix to estimate possible positions of camera
        """ 
        # see Szeliski CV 2nd Ed. pg. 707
        
        # matrix for rotating 90 degrees
        R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        
        # SVD of essential matrix
        U, S, V = np.linalg.svd(E)
        
        # singular vector associated with the smallest (right) singular value gives us translation vector
        C1 = U[:, 2].reshape(-1, 1)
        C2 = -U[:, 2].reshape(-1, 1)
        C3 = U[:, 2].reshape(-1, 1)
        C4 = -U[:, 2].reshape(-1, 1)
        
        # R = U * Rot_90_degrees * V
        # need to generate 4 possible positions 
        # U and V are not guarenteed to be rotations - can flip both signs and still get valid SVD
        # pair both rotation matrices with both possible signs of translation vector 
        R1 = U @ R_z @ V
        R2 = U @ R_z @ V
        R3 = U @ R_z.T @ V
        R4 = U @ R_z.T @ V
        
        # if the determinant of R is -1 we need to flip the sign
        if np.linalg.det(R1) < 0:
            R1 = -R1
            C1 = -C1
            
        if np.linalg.det(R2) < 0:
            R2 = -R2
            C2 = -C2

        if np.linalg.det(R3) < 0:
            R3 = -R3
            C3 = -C3
            
        if np.linalg.det(R4) < 0:
            R4 = -R4
            C4 = -C4
            
        return C1, C2, C3, C4, R1, R2, R3, R4
              
    def correct_coord(self, R, t, best_curr, best_next, k):
        """
        determine cheirality -- number of points in front of camera
        """
        # see Zisserman Multiple View Geometry CV 2nd Ed. pg. 518
        C = [[0], [0], [0]]
        R1 = np.eye(3)
        x = self.linear_triangulation(R, R1, C, t, k, best_curr, best_next)
        x = np.array(x)
        depth = 0
        
        for i in range(x.shape[0]):
            x_n = x[i, :].reshape(-1, 1)
            
            # a 3D point x is in front of the camera if r_3 * (x − t) > 0, where r_3 is the third row
            # of the rotation matrix and x = (x, y, z, 1)
            # distance from front of camera must be positive!
            if np.dot(R[2], np.subtract(x_n[:3], t)) > 0 and x_n[2] > 0:
                depth += 1
        
        return depth
        
    def linear_triangulation(self, R1, R2, C1, C2, k, best_curr, best_next):
        """
        performs simple linear triangulation
        """    
        # see Zisserman Multiple View Geometry CV 2nd Ed. pg. 312 
        # see Szeliski CV 2nd Ed. pg. 708
        
        # build projection matrices - P = K[R|t] where t = -R * C and C = camera center
        P1 = np.dot(k, np.hstack((R1, np.dot(-R1, C1))))
        P2 = np.dot(k, np.hstack((R2, np.dot(-R2, C2))))
        X = []
        
        for i in range(len(best_curr)):
            # A = [(x * p_3 - p_1), (y * p_3 - p_2), (x' * p'_3 - p'_1), (y' * p'_3 - p'_2)]
            x1 = best_curr[i]
            x2 = best_next[i]
            A1 = x1[0] * P1[2, :] - P1[0, :]
            A2 = x1[1] * P1[2, :] - P1[1, :]
            A3 = x2[0] * P2[2, :] - P2[0, :]
            A4 = x2[1] * P2[2, :] - P2[1, :]
            
            A = [A1, A2, A3, A4]
            U, S, V = np.linalg.svd(A)
            V = V[3]
            V = V / V[-1]
            X.append(V)
        
        return X
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera

        The returned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        frames = self.frames
        f = self.focal_length
        px, py = self.pp
        k = np.array([[f, 0, px], [0, f, py], [0, 0, 1]])
        path = np.zeros((len(frames), 3))
        ransac_thresh = 0.92
        
        # initial camera position matrix
        previous_pos = np.eye(4)
        
        for frame in range(len(frames)):
            
            # !! FIX -- last frame is incorrect
            if frame == len(frames) - 1:
                path[frame] = path[frame - 1] 
                break
                
            # print progress to make sure function is not stuck
            # if frame % 20 == 0:
            #    print(frame)
                
            # extract scale
            if frame == 0:
                scale = 1
            else:
                scale = self.get_scale(frame)
                
            # select current and next frame (assumes frames are in order)
            current_f = self.imread(frames[frame])
            next_f = self.imread(frames[frame + 1])
            
            # use BRISK to calculate keypoints and descriptors
            # initially used SIFT but autograder didn't have access to it
            # ORB was very inaccurate compared to SIFT and BRISK 
            brisk = cv2.BRISK_create()
            kp_current, des_current = brisk.detectAndCompute(current_f, None)
            kp_next, des_next = brisk.detectAndCompute(next_f, None)
            
            # use k=2 so we can find best matches using distance check 
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_current, des_next, k=2)
            
            best_matches = []
            
            # Lowe's ratio test -- keep matches with smallest distance
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    best_matches.append(m)
                 
            # initialize array to hold coordinates of matches for current and next frame
            best_curr = np.zeros((len(best_matches), 2))
            best_next = np.zeros((len(best_matches), 2))
            
            # extract x, y coordinates
            for i, m in enumerate(best_matches):
                best_curr[i] = kp_current[m.queryIdx].pt[0], kp_current[m.queryIdx].pt[1]
                best_next[i] = kp_next[m.trainIdx].pt[0], kp_next[m.trainIdx].pt[1]
            
            # convert to homogeneous coordinates 
            ones = np.ones((best_curr.shape[0], 1))
            new_curr = np.column_stack((best_curr, ones))
            new_next = np.column_stack((best_next, ones))
            
            new_curr = np.ascontiguousarray(new_curr)
            new_next = np.ascontiguousarray(new_next)
                
            F_init = FundamentalMatrix()
            ransac_F = Ransac(F_init)

            total_data = np.column_stack((new_curr, new_next))
            F, best_curr, best_next = ransac_F.fit(total_data, ransac_thresh)
            
            # calculate essential matrix
            E = self.essential_matrix(k, F)
            
            # get four possible camera positions (translation vectors) and rotation matrices
            C1, C2, C3, C4, R1, R2, R3, R4 = self.camera_position(E)
            
            # see Szeliski CV 2nd Ed. pg. 708
            # choose best translation vector by determining which has the largest # pts in front of both cameras
            C1_depth = self.correct_coord(R1, C1, best_curr, best_next, k)
            C2_depth = self.correct_coord(R2, C2, best_curr, best_next, k)
            C3_depth = self.correct_coord(R3, C3, best_curr, best_next, k)
            C4_depth = self.correct_coord(R4, C4, best_curr, best_next, k)
            
            T = [C1, C2, C3, C4]
            R = [R1, R2, R3, R4]
            
            # get index for largest # points
            best_ind = np.argmax([C1_depth, C2_depth, C3_depth, C4_depth])
            
            T = T[best_ind] 
            R = R[best_ind]
            
            # if z value is negative reverse sign
            if T[2] < 0: 
                T = -T
            
            # scale using ground truth value difference between current coordinate and previous coordinate
            T *= scale
            
            best_position = np.column_stack((R, T))
            best_position = np.vstack((best_position, np.array([0, 0, 0, 1])))
            
            # apply translation to obtain coordinates of camera
            previous_pos = np.dot(previous_pos, best_position)
            
            path[frame][0] = -previous_pos[0, -1] 
            path[frame][1] = previous_pos[1, -1] 
            path[frame][2] = previous_pos[2, -1]
            #print(path[frame])
        
        return path

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
