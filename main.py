import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *
import pdb
import pickle

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 101 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'
	plot_haar_filter = 'haar_filters' if not flag_subset else 'haar_filters_subset'
	plot_sc_errors = 'sc_errors' if not flag_subset else 'sc_errors_subset'
	steps = [0, 10, 50, 100] if not flag_subset else [0, 10]

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)

	#---HARD NEGATIVE MINING--
	#putting non-faces into training data for hard-negative mining
	'''for i in range(3):
		negative_patches = pickle.load(open('wrong_patches_'+str(i)+'.pkl', 'rb'))
		data = np.append(data, negative_patches, axis = 0)
		labels = np.append(labels, np.full(len(negative_patches), -1))'''
	#pdb.set_trace()
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 8 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)
	print("Length of filters " + str(len(filters)))

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type, chosen_wc_cache_dir)

	#calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	print("Start of train process")
	boost.train(chosen_wc_cache_dir)
	print("End of train process")
	
	print("Plotting Haar Filters")
	boost.display_haar_filters(chosen_wc_cache_dir, plot_haar_filter)

	print("Plotting training error of strong classifier")
	boost.draw_sc_errors(chosen_wc_cache_dir, plot_sc_errors)

	#Histogram, ROC, weak classfier errors
	boost.visualize(steps, chosen_wc_cache_dir)

	
	print("------Face Detection---------")

	original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_Face2_hardneg.png', result_img)

	original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_Face3_hardneg.png', result_img)


	#HARD NEGATIVE MINING
	'''
	print("------Hard Negative Mining---------")
	image_names = ['Non_face_1', 'Non_Face_2', 'Non_face_3']
	for img in image_names:
		print('Testing_Images/' + img + '.jpg')
    wrong_patches = []
	for img in image_names:
		original_img = cv2.imread('Testing_Images/' + img + '.jpg', cv2.IMREAD_GRAYSCALE)
		wrong_patches.append(boost.get_hard_negative_patches(original_img))
    wrong_patches_0 = wrong_patches[0].reshape(wrong_patches[0].shape[1:4])
	wrong_patches_1 = wrong_patches[1].reshape(wrong_patches[1].shape[1:4])
	wrong_patches_2 = wrong_patches[2].reshape(wrong_patches[2].shape[1:4])
	pickle.dump(wrong_patches_0, open( 'wrong_patches_0.pkl', 'wb'))
	pickle.dump(wrong_patches_1, open( 'wrong_patches_1.pkl', 'wb'))
	pickle.dump(wrong_patches_2, open( 'wrong_patches_2.pkl', 'wb'))
	'''
	

if __name__ == '__main__':
	main()
