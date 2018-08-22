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
        self.vr_paths = []
        self.vr_indices = []

        self.rb_paths = []
        self.rb_indices = []


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
        #  if FLAGS.train:
            #  # Normalize the states if it's training.
            #  with Timer('Normalizing states'):
                #  if self.scale is None or self.bias is None:
                    #  #for i in xrange(len(demos)):
                    #  #    print len(demos[i]['demoX'])
                    #  #    print self.allepisodes.path[i]
                    #  states = np.vstack((demos[i]['demoX'] for i in xrange(len(demos)))) # hardcoded here to solve the memory issue
                    #  states = states.reshape(-1, len(self.state_idx))
                    #  # 1e-3 to avoid infs if some state dimensions don't change in the
                    #  # first batch of samples
                    #  self.scale = np.diag(
                        #  1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                    #  self.bias = - np.mean(
                        #  states.dot(self.scale), axis=0)
                    #  # Save the scale and bias.
                    #  with open('data/scale_and_bias_%s.pkl' % FLAGS.experiment, 'wb') as f:
                        #  pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
                #  for key in xrange(len(demos)):
                    #  self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.state_idx))
                    #  self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                    #  self.allepisodes.traj[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.state_idx))


        self.alltestepisodes = NEEM()
        #self.alltestepisodes.indices, self.alltestepisodes.paths = self.sample_idx(FLAGS.end_test_set_size, FLAGS.number_of_shot_test, 0)

        # generate episode batches for training
        if FLAGS.train:
            self.alltrainepisodes = Sample()
            self.allvalidationepisodes = Sample()

            for itr in xrange(self.total_iters):
                # 'sample_idx' returns : vr_samples, vr_paths, rb_samples, rb_paths
                vr_itr_data, vr_itr_path, rb_itr_data, rb_itr_path = self.sample_idx(self.meta_batch_size, self.number_of_shot, self.test_batch_size)
                self.alltrainepisodes.vr_indices.extend(vr_itr_data)
                self.alltrainepisodes.vr_paths.extend(vr_itr_path)
                self.alltrainepisodes.rb_indices.extend(rb_itr_data)
                self.alltrainepisodes.rb_paths.extend(rb_itr_path)

                #if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                if itr % TEST_PRINT_INTERVAL == 0:
                    vr_val_data, vr_val_path, rb_val_data, rb_val_path = self.sample_idx(self.meta_batch_size, self.number_of_shot, self.test_batch_size)
                    self.allvalidationepisodes.vr_indices.extend(vr_val_data)
                    self.allvalidationepisodes.vr_paths.extend(vr_val_path)
                    self.allvalidationepisodes.rb_indices.extend(rb_val_data)
                    self.allvalidationepisodes.rb_paths.extend(rb_val_path)

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
            experiment_subfolder = set(natsorted(glob.glob(folders[idx] + '/*')))
            rb_experiment_subfolder = set(natsorted(glob.glob(folders[idx] + '/*-Robot*')))
            vr_experiment_subfolder = list(experiment_subfolder - rb_experiment_subfolder)
            rb_experiment_subfolder = list(rb_experiment_subfolder)

            #  subrange_exp = len(experiment_subfolder)
            vr_subrange_exp = len(vr_experiment_subfolder)
            rb_subrange_exp = len(rb_experiment_subfolder)

            episode_paths = []
            demos=dict([('demoX', []), ('demoU', [])])

            # for vr only
            for ind in xrange(vr_subrange_exp): # Task-HCP-BPB-0
                #current_path = folders[idx] + '/' + experiment_subfolder[ind]
                current_path = vr_experiment_subfolder[ind]
                for fname in os.listdir(current_path):
                    if fname.endswith('.txt'):
                        #episode_paths.append(current_path + '/imgs/animation.gif')
                        self.extract_txt(current_path)
                        current_samples = self.extract_experiment_data(current_path + "/" + fname)

                        # Asil's idea
                        for t in range(1, len(current_samples['demoU'])):
                            current_samples['demoX'][t-1] = current_samples['demoU'][t]

                        current_samples['demoX'][15] = current_samples['demoU'][15]

                        demos['demoU'].append(current_samples['demoU'])
                        demos['demoX'].append(current_samples['demoX'])

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

            for ind in xrange(rb_subrange_exp):
                current_path = rb_experiment_subfolder[ind]

                for fname in os.listdir(current_path):
                    if fname.endswith('.txt'):
                        #episode_paths.append(current_path + '/imgs/animation.gif')
                        self.extract_txt(current_path)
                        current_samples = self.extract_experiment_data(current_path + "/" + fname)
                        demos['demoU'].append(current_samples['demoU'])
                        demos['demoX'].append(current_samples['demoX'])

                        episode_paths.append(current_path + '/imgs/animation.gif')

            demos['demoU'] = np.array(demos['demoU'])
            demos['demoX'] = np.array(demos['demoX'])
            path.append(episode_paths)
            traj.append(demos)
        return path, traj



    def sample_idx(self, batch_size, update_batch_size, test_batch_size): # update_batch_size: num_of_shot
        range_exp = range(len(self.allepisodes.traj))

        sampled_episode_folders = random.sample(range_exp, batch_size)
        vr_samples = []
        vr_paths = []
        rb_samples = []
        rb_paths = []
        sampled_subepisodes = []
        for idx in sampled_episode_folders:
            _subrange_len = self.allepisodes.traj[idx]['demoX'].shape[0]
            vr_subrange_exp = range(_subrange_len-1)
            rb_subrange_exp = range(_subrange_len-1, _subrange_len)

            vr_sampled_subepisodes = random.sample(vr_subrange_exp, update_batch_size)
            rb_sampled_subepisodes = random.sample(rb_subrange_exp, test_batch_size)

            for ind in vr_sampled_subepisodes:
                vr_samples.append([idx,ind])
                vr_paths.append(self.allepisodes.paths[idx][ind])

            for ind in rb_sampled_subepisodes:
                rb_samples.append([idx,ind])
                rb_paths.append(self.allepisodes.paths[idx][ind])

        return vr_samples, vr_paths, rb_samples, rb_paths


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
        vr_batch_image_size = (self.number_of_shot) * self.meta_batch_size
        rb_batch_image_size = (self.test_batch_size) * self.meta_batch_size

        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']

        if train:
            vr_filenames = self.alltrainepisodes.vr_paths
            rb_filenames = self.alltrainepisodes.rb_paths
        else:
            vr_filenames = self.allvalidationepisodes.vr_paths
            rb_filenames = self.allvalidationepisodes.rb_paths

        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        # make queue for tensorflow to read from"
        vr_filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(vr_filenames), shuffle=False)
        rb_filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(rb_filenames), shuffle=False)

        vr_image_reader = tf.WholeFileReader()
        rb_image_reader = tf.WholeFileReader()

        _, vr_image_file = vr_image_reader.read(vr_filename_queue)
        _, rb_image_file = rb_image_reader.read(rb_filename_queue)

        vr_image = tf.image.decode_gif(vr_image_file)
        rb_image = tf.image.decode_gif(rb_image_file)

        # should be T x C x W x H
        vr_image.set_shape((self.T, im_height, im_width, num_channels))
        rb_image.set_shape((self.T, im_height, im_width, num_channels))

        vr_image = tf.cast(vr_image, tf.float32)
        rb_image = tf.cast(rb_image, tf.float32)
        vr_image /= 255.0
        rb_image /= 255.0

        vr_image = tf.transpose(vr_image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        vr_image = tf.reshape(vr_image, [self.T, -1])

        rb_image = tf.transpose(rb_image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        rb_image = tf.reshape(rb_image, [self.T, -1])

        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 64 #128 #256
        print 'Batching images'
        vr_images = tf.train.batch(
                [vr_image],
                batch_size = vr_batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * vr_batch_image_size,
                )
        rb_images = tf.train.batch(
                [vr_image],
                batch_size = rb_batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * rb_batch_image_size,
                )


        vr_all_images = []
        for i in xrange(self.meta_batch_size):
            image = vr_images[i*(self.number_of_shot):(i+1)*(self.number_of_shot)]
            image = tf.reshape(image, [(self.number_of_shot)*self.T, -1])
            vr_all_images.append(image)

        rb_all_images = []
        for i in xrange(self.meta_batch_size):
            image = rb_images[i*(self.test_batch_size):(i+1)*(self.test_batch_size)]
            image = tf.reshape(image, [(self.test_batch_size)*self.T, -1])
            rb_all_images.append(image)

        return tf.stack(vr_all_images), tf.stack(rb_all_images)

    def generate_data_batch(self, itr, train=True):

        batch_size = self.meta_batch_size                       # number of tasks sampled per meta-update
        update_batch_size = self.number_of_shot                 # number of examples used for inner gradient update
        test_batch_size = self.test_batch_size                  # number of examples used for gradient update during training

        demos = self.allepisodes.traj
        if train:
            vr_indices = self.alltrainepisodes.vr_indices[batch_size*itr*(self.number_of_shot):batch_size*(itr+1)*(self.number_of_shot)]
            rb_indices = self.alltrainepisodes.rb_indices[batch_size*itr*(self.test_batch_size):batch_size*(itr+1)*(self.test_batch_size)]
        else:
            vr_indices = self.allvalidationepisodes.vr_indices[batch_size*itr*(self.number_of_shot):batch_size*(itr+1)*(self.number_of_shot)]
            rb_indices = self.allvalidationepisodes.rb_indices[batch_size*itr*(self.test_batch_size):batch_size*(itr+1)*(self.test_batch_size)]

        vr_demo_size = len(vr_indices)
        rb_demo_size = len(rb_indices)
        vr_U = []
        vr_X = []
        rb_U = []
        rb_X = []

        for i in xrange(vr_demo_size):
            vr_U.append(demos[vr_indices[i][0]]['demoU'][vr_indices[i][1]])
            vr_X.append(demos[vr_indices[i][0]]['demoX'][vr_indices[i][1]])
        for i in xrange(rb_demo_size):
            rb_U.append(demos[rb_indices[i][0]]['demoU'][rb_indices[i][1]])
            rb_X.append(demos[rb_indices[i][0]]['demoX'][rb_indices[i][1]])

        vr_U = np.array(vr_U)
        vr_X = np.array(vr_X)

        rb_U = np.array(rb_U)
        rb_X = np.array(rb_X)

        vr_U = vr_U.reshape(batch_size, (update_batch_size)*self.T, -1)
        vr_X = vr_X.reshape(batch_size, (update_batch_size)*self.T, -1)
        rb_U = rb_U.reshape(batch_size, (test_batch_size)*self.T, -1)
        rb_X = rb_X.reshape(batch_size, (test_batch_size)*self.T, -1)

        return vr_X, vr_U, rb_U, rb_X
