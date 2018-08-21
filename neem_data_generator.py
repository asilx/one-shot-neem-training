""" Code for extracting data from NEEMs and batching them together with images. Developer: Asil Kaan Bozcuoglu """
from __future__ import division
import re
import os
import readline
from json_prolog_commandline import PQ

import logging
import os.path
import glob
import pickle

import numpy as np
import math
import random
import tensorflow as tf
from utils import extract_demo_dict, Timer
from tensorflow.python.platform import flags
from natsort import natsorted
from random import shuffle
from PIL import Image
import imageio

FLAGS = flags.FLAGS


class NEEM(object):
    def __init__(self, config={}):
        self.paths = []
        self.traj = []


class Sample(object):
    def __init__(self, config={}):
        self.paths = []
        self.indices = []


class NEEMDataGenerator(object):
    def __init__(self, config={}):
        # Hyperparameters
        self.number_of_shot = FLAGS.number_of_shot
        self.test_batch_size = FLAGS.number_of_shot_train if FLAGS.number_of_shot_train != -1 else self.number_of_shot
        self.meta_batch_size = FLAGS.meta_batch_size
        self.T = FLAGS.TimeFrame

        self.total_iters = FLAGS.metatrain_iterations
        TEST_PRINT_INTERVAL = 500

        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        # NEEMs related
        self.neems = os.path.join( FLAGS.data_path, 'others')
        self.path2vrdemo = os.path.join( FLAGS.data_path, 'low_res_data')

        self.prologquery = FLAGS.featurizequery
        self.initquery = FLAGS.initquery
        self.retractquery = FLAGS.retractquery
        self.parsequery = FLAGS.parsequery
        self.timequery = FLAGS.timequery

        prologinstance = PQ()
        prologinstance.prolog_query(self.initquery)

        experiment_folders = natsorted(glob.glob(self.path2vrdemo + '/*'))
        experiment_subfolder_lengths = []

        self.allepisodes = NEEM()

        self.allepisodes.paths, self.allepisodes.traj = self.bring_episodes_to_memory(experiment_folders)
        demos = self.allepisodes.traj
        self.state_idx = range(demos[0]['demoX'].shape[-1])
        self._dU = demos[0]['demoU'].shape[-1]
        if FLAGS.train:
            # Normalize the states if it's training.
            with Timer('Normalizing states'):
                if self.scale is None or self.bias is None:
                    #for i in xrange(len(demos)):
                    #    print len(demos[i]['demoX'])
                    #    print self.allepisodes.path[i]
                    states = np.vstack((demos[i]['demoX'] for i in xrange(len(demos)))) # hardcoded here to solve the memory issue
                    states = states.reshape(-1, len(self.state_idx))
                    # 1e-3 to avoid infs if some state dimensions don't change in the
                    # first batch of samples
                    self.scale = np.diag(
                        1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                    self.bias = - np.mean(
                        states.dot(self.scale), axis=0)
                    # Save the scale and bias.
                    with open('data/scale_and_bias_%s.pkl' % FLAGS.experiment, 'wb') as f:
                        pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
                for key in xrange(len(demos)):
                    self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.state_idx))
                    self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                    self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.state_idx))


        self.alltestepisodes = NEEM()
        #self.alltestepisodes.indices, self.alltestepisodes.paths = self.sample_idx(FLAGS.end_test_set_size, FLAGS.number_of_shot_test, 0)

        # generate episode batches for training
        if FLAGS.train:
            self.alltrainepisodes = Sample()
            self.allvalidationepisodes = Sample()

            for itr in xrange(self.total_iters):
                itr_data, itr_path = self.sample_idx(self.meta_batch_size, self.number_of_shot, self.test_batch_size)
                self.alltrainepisodes.indices.extend(itr_data)
                self.alltrainepisodes.paths.extend(itr_path)

                #if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                if itr % TEST_PRINT_INTERVAL == 0:
                    val_data, val_path = self.sample_idx(self.meta_batch_size, self.number_of_shot, self.test_batch_size)
                    self.allvalidationepisodes.indices.extend(val_data)
                    self.allvalidationepisodes.paths.extend(val_path)

    def create_sub_gif(self, path, targetpath, start, end):
        frame = Image.open(path)
        images = []
        for x in range(int(3 * math.floor(start)), (int(3* math.floor(end)) + 1)):
            try:
                frame.seek(x)
                new_im = Image.new("RGB", frame.size)
                new_im.paste(frame)
                images.append(np.array(new_im))
            except EOFError:
                break;
        if len(images) < self.T:
            print 'Subgif frame count is less than 16. Fixing it'
            self.create_sub_gif(path, targetpath, start - 5, end - 5)
        else:
            imageio.mimsave(targetpath, images)

    def bring_episodes_to_memory(self, folders):
        range_exp = len(folders)
        path = []
        traj = []
        for idx in xrange(range_exp):
            experiment_subfolder = natsorted(glob.glob(folders[idx] + '/*'))
            subrange_exp = len(experiment_subfolder)
            episode_paths = []
            demos=dict([('demoX', []), ('demoU', [])])
            for ind in xrange(subrange_exp): # Task-HCP-BPB-0
                #current_path = folders[idx] + '/' + experiment_subfolder[ind]
                current_path = experiment_subfolder[ind]
                for fname in os.listdir(current_path):
                    if fname.endswith('.txt'):
                        #episode_paths.append(current_path + '/imgs/animation.gif')
                        self.extract_txt(current_path)
                        current_samples = self.extract_experiment_data(current_path + "/" + fname)
                        demos['demoU'].append(current_samples['demoU'])
                        demos['demoX'].append(current_samples['demoX'])
                        #current_samples['demoU'] = np.array(current_samples['demoU'])
                        #current_samples['demoX'] = np.array(current_samples['demoX'])
                        if FLAGS.train:
                            retractinstance = PQ()
                            retractinstance.prolog_query(self.retractquery)
                            neem_path = os.path.join(self.neems, os.path.basename(current_path), 'v0/log.owl')
                            parseinstance = PQ()
                            parseinstance.prolog_query(self.parsequery % neem_path)
                            timeinstance = PQ()
                            taskname = os.path.basename(fname).replace('.txt', '')
                            solutions = timeinstance.prolog_query(self.timequery % taskname)
                            endtime = -1
                            for s in solutions:
                                for k, v in s.items():
                                    endtime = v
                            starttime = endtime - 6
                            endtime = endtime - 1
                            print endtime
                            targetpath = current_path + '/imgs/' + taskname + '.gif'
                            self.create_sub_gif(current_path + '/imgs/animation.gif', targetpath, starttime, endtime)
                            episode_paths.append(targetpath)
                        else:
                            episode_paths.append(current_path + '/imgs/animation.gif')
                        #path.append(targetpath)
                        #traj.append(current_samples)
            demos['demoU'] = np.array(demos['demoU'])
            demos['demoX'] = np.array(demos['demoX'])
            path.append(episode_paths)
            traj.append(demos)
        return path, traj

    def sample_idx(self, batch_size, update_batch_size, test_batch_size):
        range_exp = range(len(self.allepisodes.traj))
        sampled_episode_folders = random.sample(range_exp, batch_size)
        samples = []
        paths = []
        sampled_subepisodes = []
        for idx in sampled_episode_folders:
            subrange_exp = range(self.allepisodes.traj[idx]['demoX'].shape[0])
            sampled_subepisodes = random.sample(subrange_exp, update_batch_size + test_batch_size)
            for ind in sampled_subepisodes:
                samples.append([idx,ind])
                paths.append(self.allepisodes.paths[idx][ind])
        return samples, paths

    def extract_txt(self, txtdirectory):
        feature_file_exist = False
        for fname in os.listdir(txtdirectory):
            if fname.endswith('.txt'):
                feature_file_exist = True
                break
        if not feature_file_exist:
            prologquery = self.prologquery % (self.neems, txtdirectory)
            prologinstance = PQ()
            prologinstance.prolog_query(prologquery)

    def extract_experiment_data(self, txt_file_path):
        content=dict([('demoX', []), ('demoU', [])])
        with open(txt_file_path, 'r') as f:
            data = f.readlines()
        velocity_part = False
        for line in data:
            if line == '---\n':
                velocity_part = True
                continue
            else:
                chunk = line.split(',')
                sample = []
                sample.append(float(chunk[0]))
                sample.append(float(chunk[1]))
                sample.append(float(chunk[2]))
                sample.append(float(chunk[3]))
                sample.append(float(chunk[4]))
                sample.append(float(chunk[5]))
                sample.append(float(chunk[6]))
            if velocity_part == False:
                content['demoX'].append(np.array(sample))
            else:
                content['demoU'].append(np.array(sample))
        return content

    def make_batch_tensor(self, network_config, train=True):
        batch_image_size = (self.number_of_shot + self.test_batch_size) * self.meta_batch_size

        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']

        if train:
            all_filenames = self.alltrainepisodes.paths
        else:
            all_filenames = self.allvalidationepisodes.paths

        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        # make queue for tensorflow to read from"
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)

        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.T, im_height, im_width, num_channels))
        image = tf.cast(image, tf.float32)
        image /= 255.0

        image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 64 #128 #256
        print 'Batching images'
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []
        for i in xrange(self.meta_batch_size):
            image = images[i*(self.number_of_shot+self.test_batch_size):(i+1)*(self.number_of_shot+self.test_batch_size)]
            image = tf.reshape(image, [(self.number_of_shot+self.test_batch_size)*self.T, -1])
            all_images.append(image)
        return tf.stack(all_images)

    def generate_data_batch(self, itr, train=True):

        batch_size = self.meta_batch_size                       # number of tasks sampled per meta-update
        update_batch_size = self.number_of_shot                 # number of examples used for inner gradient update
        test_batch_size = self.test_batch_size                  # number of examples used for gradient update during training

        demos = self.allepisodes.traj
        if train:
            indices = self.alltrainepisodes.indices[batch_size*itr*(self.number_of_shot+self.test_batch_size):batch_size*(itr+1)*(self.number_of_shot+self.test_batch_size)]
        else:
            indices = self.allvalidationepisodes.indices[batch_size*itr*(self.number_of_shot+self.test_batch_size):batch_size*(itr+1)*(self.number_of_shot+self.test_batch_size)]

        demo_size = len(indices)
        U = []
        X = []

        for i in xrange(demo_size):
            U.append(demos[indices[i][0]]['demoU'][indices[i][1]])
            X.append(demos[indices[i][0]]['demoX'][indices[i][1]])

        U = np.array(U)
        X = np.array(X)

        U = U.reshape(batch_size, (test_batch_size+update_batch_size)*self.T, -1)
        X = X.reshape(batch_size, (test_batch_size+update_batch_size)*self.T, -1)

        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.state_idx)
        return X, U
