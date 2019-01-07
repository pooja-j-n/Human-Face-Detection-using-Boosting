import datetime
import logging as log
import math
import matplotlib.pyplot as plt 
from operator import itemgetter
from copy import deepcopy
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize

import pdb

log.basicConfig(filename='./train.log', level=log.DEBUG)


class Boosting_Classifier:
    def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style, save_dir):
        self.filters = haar_filters
        self.data = data
        self.labels = labels
        self.num_chosen_wc = num_chosen_wc
        self.num_bins = num_bins
        self.visualizer = visualizer
        self.num_cores = num_cores
        self.style = style
        self.chosen_wcs = None
        if style == 'Ada':
            self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     for i, filt in enumerate(self.filters)]
        elif style == 'Real':
            if save_dir is not None and os.path.exists(save_dir):
                print('[Loading chosen weak classifiers from Adaboost, %s loading...]' % save_dir)
                self.load_trained_wcs(save_dir)
            else:
                print("Chosen classifiers not found")
                return

            self.chosen_classifiers_from_AB = np.array(self.chosen_wcs)[:,1]

            self.weak_classifiers = [
                Real_Weak_Classifier(i, self.chosen_classifiers_from_AB[i].plus_rects, self.chosen_classifiers_from_AB[i].minus_rects,
                                     self.num_bins) for i in range(len(self.chosen_classifiers_from_AB))]


    def calculate_training_activations(self, save_dir=None, load_dir=None):
        print(load_dir)
        print('Calcuate activations for %d weak classifiers, using %d imags.' % (
        len(self.weak_classifiers), self.data.shape[0]))
        if self.style=='Real':
            for i in range(len(self.weak_classifiers)):
                self.weak_classifiers[i].activations = self.chosen_classifiers_from_AB[i].activations
            return
        if load_dir is not None and os.path.exists(load_dir):
            print('[Find cached activations, %s loading...]' % load_dir)
            wc_activations = np.load(load_dir)
        else:
            if self.num_cores == 1:
                wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
            else:
                wc_activations = Parallel(n_jobs=self.num_cores)(
                    delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
            wc_activations = np.array(wc_activations)
            if save_dir is not None:
                print('Writing results to disk...')
                np.save(save_dir, wc_activations)
                print('[Saved calculated activations to %s]' % save_dir)
        for wc in self.weak_classifiers:
            wc.activations = wc_activations[wc.id, :]
        return wc_activations

    # select weak classifiers to form a strong classifier
    # after training, by calling self.sc_function(), a prediction can be made
    # self.chosen_wcs should be assigned a value after self.train() finishes
    # call Weak_Classifier.calc_error() in this function
    # cache training results to self.visualizer for visualization
    #
    #
    # detailed implementation is up to you
    # consider caching partial results and using parallel computing
    def train(self, save_dir=None):

        if self.style=='Ada':

            if save_dir is not None and os.path.exists(save_dir):
                print('[Find cached weak classifiers, %s loading...]' % save_dir)
                self.load_trained_wcs(save_dir)
                return

            print("Training------------")

            self.chosen_wcs = []
            size_data = len(self.data)
            data_point_weights = np.full((size_data), 1 / size_data)
            T = self.num_chosen_wc
            for t in range(T):
                print(str(datetime.datetime.now()) + "T------------" + str(t))
                if self.num_cores == 1:
                    wc_output = [wc.calc_error(data_point_weights, self.labels) for wc in self.weak_classifiers]
                else:
                    print("Parallelizing")
                    wc_output = Parallel(n_jobs=self.num_cores)(delayed(wc.calc_error)(data_point_weights, self.labels) for wc in self.weak_classifiers)
                # wc_errors = Parallel(n_jobs=self.num_cores)(delayed(wc.calc_error)(data_point_weights, self.labels) for wc in self.weak_classifiers)
                wc_output = np.array(wc_output)
                min_error_index = np.argmin(wc_output[:,0])
                min_error = wc_output[min_error_index][0]
                if t==0 or t==10 or t==50 or t==100:
                    print("Saving top 1000 errors ")
                    pickle.dump(np.sort(wc_output[:,0])[:1000], open("top_errors_" + str(t), "wb"))
                '''log.info(str(datetime.datetime.now()) + "Minimum Error"  + str(min_error))
                log.info(str(datetime.datetime.now()) + "Minimum Error Index"  + str(min_error_index))'''

                alpha_best_wc = 0.5 * np.log((1 - min_error) / min_error)
                '''log.info(str(datetime.datetime.now()) + "alpha" + str(alpha_best_wc))'''

                best_wc = deepcopy(self.weak_classifiers[min_error_index])
                best_wc.threshold = wc_output[min_error_index][1]
                best_wc.polarity = wc_output[min_error_index][2]

                self.chosen_wcs.append((alpha_best_wc, best_wc))

                data_point_weights = data_point_weights * [np.exp(-1 * alpha_best_wc * self.labels[dp] * (
                            best_wc.polarity * np.sign(best_wc.activations[dp] - best_wc.threshold))) for dp in
                                                           range(size_data)]
                data_point_weights = data_point_weights / sum(data_point_weights)

                train_predicts = []
                for idx in range(self.data.shape[0]):
                    train_predicts.append(self.sc_function(self.data[idx, ...]))
                print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
                print("Updated weights")

            ######################
            ######## TODO ########
            ######################
            if save_dir is not None:
                pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

        elif self.style=='Real':
            print("Training------------")


            if save_dir is not None and os.path.exists(save_dir):
                print('[Loading chosen weak classifiers from Adaboost, %s loading...]' % save_dir)
                self.load_trained_wcs(save_dir)
            else:
                print("Chosen classifiers not found")
                return

            chosen_classifiers = self.weak_classifiers
            self.chosen_wcs_real = []
            size_data = len(self.data)
            T = self.num_chosen_wc
            data_point_weights = np.full((size_data), 1 / size_data)
            for t in range(T):
                print(str(datetime.datetime.now()) + "T------------" + str(t))
                chosen_classifiers[t].calc_error(data_point_weights, self.labels, t)
                chosen_classifier_real = deepcopy(chosen_classifiers[t])
                self.chosen_wcs_real.append(chosen_classifier_real)


                responses = []
                for i in range(size_data):
                    bin_idx = np.sum(chosen_classifier_real.thresholds < chosen_classifier_real.activations[i])
                    responses.append(0.5 * np.log(chosen_classifier_real.bin_pqs[0, bin_idx] / chosen_classifier_real.bin_pqs[1, bin_idx]))

                if (np.count_nonzero(chosen_classifier_real.bin_pqs) != 2*chosen_classifier_real.num_bins):
                    print("-------------------Zeros in ps and qs--------------------")
                #pdb.set_trace()
                train_predicts = []
                for idx in range(size_data):
                    #pdb.set_trace()
                    train_predicts.append(np.sum([np.array([wc.predict_image(self.data[idx, ...]) for wc in self.chosen_wcs_real])]))

                #pdb.set_trace()
                print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))

                data_point_weights = data_point_weights * np.exp(-1 * self.labels * responses)

                data_point_weights = data_point_weights/np.sum(data_point_weights)

                #pdb.set_trace()

            #pdb.set_trace()
            save_dir = save_dir.replace("chosen_wcs", "chosen_wcs_real")
            if save_dir is not None:
                pickle.dump(self.chosen_wcs_real, open(save_dir, 'wb'))





    def sc_function(self, image):
        if self.style=='Ada':
            return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])
        else:
            return np.sum([np.array([wc.predict_image(image) for wc in self.chosen_wcs_real])])


    def load_trained_wcs(self, save_dir):
        self.chosen_wcs = pickle.load(open(save_dir, 'rb'))

    def face_detection(self, img, scale_step=20):

        # this training accuracy should be the same as your training process,
        ##################################################################################
        train_predicts = []
        for idx in range(self.data.shape[0]):
            train_predicts.append(self.sc_function(self.data[idx, ...]))
        print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
        ##################################################################################

        scales = 1 / np.linspace(1, 8, scale_step)
        patches, patch_xyxy = image2patches(scales, img)
        print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
        predicts = [self.sc_function(patch) for patch in tqdm(patches)]
        print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
        pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
        if pos_predicts_xyxy.shape[0] == 0:
            return
        xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)

        print('after nms:', xyxy_after_nms.shape[0])
        for idx in range(xyxy_after_nms.shape[0]):
            pred = xyxy_after_nms[idx, :]
            cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

        return img

    def get_hard_negative_patches(self, img, scale_step=10):
        scales = 1 / np.linspace(1, 8, scale_step)
        patches, patch_xyxy = image2patches(scales, img)
        print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
        predicts = [self.sc_function(patch) for patch in tqdm(patches)]

        wrong_patches = patches[np.where(np.array(predicts) > 0), ...]

        return wrong_patches

    def display_haar_filters(self, source = None, dest = None):
        if source is not None and os.path.exists(source):
            print('[Loading cached weak classifiers, %s loading...]' % source)
            chosen_wcs = pickle.load(open(source, "rb"))
            fig, axes = plt.subplots(4, 5, figsize=(20, 10), squeeze = False)
            axes[0][0].invert_yaxis() 
            for i in range(5):
                pos_rect = chosen_wcs[i][1].plus_rects
                neg_rect = chosen_wcs[i][1].minus_rects
                axes[0][i].add_patch(plt.Rectangle((pos_rect[0][0], pos_rect[0][1]), pos_rect[0][2] - pos_rect[0][0] + 1, pos_rect[0][3] - pos_rect[0][1] + 1, fc='white', edgecolor = 'black'))
                axes[0][i].add_patch(plt.Rectangle((neg_rect[0][0], neg_rect[0][1]), neg_rect[0][2] - neg_rect[0][0] + 1, neg_rect[0][3] - neg_rect[0][1] + 1, fc='black', edgecolor = 'black'))
                axes[0][i].set_xlim([0, 16])
                axes[0][i].set_ylim([0, 16])
                pos_rect = chosen_wcs[i + 5][1].plus_rects
                neg_rect = chosen_wcs[i + 5][1].minus_rects
                axes[1][i].add_patch(plt.Rectangle((pos_rect[0][0] + 0.5, pos_rect[0][1] + 0.5), pos_rect[0][2] - pos_rect[0][0] + 1, pos_rect[0][3] - pos_rect[0][1] + 1, fc='white', edgecolor = 'black'))
                axes[1][i].add_patch(plt.Rectangle((neg_rect[0][0] + 0.5, neg_rect[0][1] + 0.5), neg_rect[0][2] - neg_rect[0][0] + 1, neg_rect[0][3] - neg_rect[0][1] + 1, fc='black', edgecolor = 'black'))
                axes[1][i].set_xlim([0, 16])
                axes[1][i].set_ylim([0, 16])
                pos_rect = chosen_wcs[i + 10][1].plus_rects
                neg_rect = chosen_wcs[i + 10][1].minus_rects
                axes[2][i].add_patch(plt.Rectangle((pos_rect[0][0] + 0.5, pos_rect[0][1] + 0.5), pos_rect[0][2] - pos_rect[0][0] + 1, pos_rect[0][3] - pos_rect[0][1] + 1, fc='white', edgecolor = 'black'))
                axes[2][i].add_patch(plt.Rectangle((neg_rect[0][0] + 0.5, neg_rect[0][1] + 0.5), neg_rect[0][2] - neg_rect[0][0] + 1, neg_rect[0][3] - neg_rect[0][1] + 1, fc='black', edgecolor = 'black'))
                axes[2][i].set_xlim([0, 16])
                axes[2][i].set_ylim([0, 16])
                pos_rect = chosen_wcs[i + 15][1].plus_rects
                neg_rect = chosen_wcs[i + 15][1].minus_rects  
                axes[3][i].add_patch(plt.Rectangle((pos_rect[0][0] + 0.5, pos_rect[0][1] + 0.5), pos_rect[0][2] - pos_rect[0][0] + 1, pos_rect[0][3] - pos_rect[0][1] + 1, fc='white', edgecolor = 'black'))
                axes[3][i].add_patch(plt.Rectangle((neg_rect[0][0] + 0.5, neg_rect[0][1] + 0.5), neg_rect[0][2] - neg_rect[0][0] + 1, neg_rect[0][3] - neg_rect[0][1] + 1, fc='black', edgecolor = 'black'))
                axes[3][i].set_xlim([0, 16])
                axes[3][i].set_ylim([0, 16])
            fig.suptitle('Top 20 Haar Filters after Adaboost training')     
            print("Saving the Top 20 Haar Filters to----- " + dest)
            #Saving the top 20 Haar Filters
            plt.savefig(dest)
            #Displaying alpha
            print("Voting Weights of Weak Classifiers")
            for i in range(20):
                print('Alpha for Weak Classifier {0} : {1}'.format(i+1, chosen_wcs[i][0]))
            plt.close()


    def visualize(self, steps, source = None):
        if source is None or not(os.path.exists(source)):
            return
        if self.style=='Real':
            source = source.replace("chosen_wcs", "chosen_wcs_real")
        chosen_wcs = pickle.load(open(source, 'rb'))
        self.visualizer.labels = self.labels
        scores = {}
        if self.style == 'Ada':
            for i in steps:
                scores[i] = np.sum(np.array([alpha * wc.polarity * np.sign(wc.activations - wc.threshold) for alpha, wc in chosen_wcs[:i + 1]]), axis = 0)
        else:
            for i in steps:
                bin_nums = np.array([np.digitize(wc.activations, wc.thresholds, right = True) for wc in chosen_wcs[:i + 1]])
                op = [0.5 * np.log(wc.bin_pqs[0, bin_nums[idx]] / wc.bin_pqs[1, bin_nums[idx]]) for idx, wc in enumerate(chosen_wcs[:i + 1])]
                scores[i] = np.sum(np.array([0.5 * np.log(wc.bin_pqs[0, bin_nums[idx]] / wc.bin_pqs[1, bin_nums[idx]]) for idx, wc in enumerate(chosen_wcs[:i + 1])]), axis = 0)
                
        self.visualizer.strong_classifier_scores = scores
        self.visualizer.boosting_type = self.style
        self.visualizer.draw_histograms()
        self.visualizer.draw_rocs()

        #Only for Adaboost
        wc_accuracy = {}
        for i in steps:
            wc_accuracy[i] = pickle.load(open('top_errors_' + str(i), 'rb')) 
        self.visualizer.weak_classifier_accuracies = wc_accuracy
        self.visualizer.draw_wc_accuracies()

    def draw_sc_errors(self, source = None, dest = None):
        if source is not None and os.path.exists(source):
            print('[Loading cached weak classifiers, %s loading...]' % source)
            chosen_wcs = pickle.load(open(source, "rb"))
            sc_errors = []
            for i in range(1, self.num_chosen_wc + 1):
                chosen_classifiers = chosen_wcs[:i]
                response = np.sign(np.sum(np.array([alpha * wc.polarity * np.sign(wc.activations - wc.threshold) for alpha, wc in chosen_classifiers]), axis=0))
                sc_errors.append(np.sum((self.labels != np.sign(response)).astype(int)))
            plt.plot([i for i in range(1, self.num_chosen_wc + 1)], sc_errors)
            plt.title('Training Errors of Strong Classifier')
            plt.xlabel('Step Number')
            plt.ylabel('Training Error')
            print("Saving the Training Errors of Strong Classifier to----- " + dest)
            plt.savefig(dest)
            plt.close()
        
