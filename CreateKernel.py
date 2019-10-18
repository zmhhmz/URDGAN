#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:41:48 2019

@author: zhouminghao
"""

import tensorflow as tf
from ops import Conv,instance_norm#,Conv_transpose
import numpy as np
from data_loader import dataloader
from tensorflow.contrib.layers import flatten
import tensorflow_utils as tf_utils

class Create_Kernel():
    
    def __init__(self, args):
        
        self.data_loader = dataloader(args)
        
        self.channel = args.channel
        self.n_feats = args.n_feats
        self.in_memory = args.in_memory
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.num_of_down_scale = args.num_of_down_scale
        self.Kgen_resblocks = args.Kgen_resblocks
        self.Kdiscrim_blocks = args.Kdiscrim_blocks
        self.vgg_path = args.vgg_path
        self.Kernel_size = args.Kernel_size
        
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
    
#    def Kgenerator(self, x, reuse = False, name = 'Kgenerator'):
#        
#        with tf.variable_scope(name_or_scope = name, reuse = reuse):
#            #_res = x
#            #x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
#            x = Conv(name = 'conv1', x = x, filter_size = 3, in_filters = self.channel, out_filters = self.n_feats, strides = 1, padding = 'SAME')
#            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
#            x = tf.nn.relu(x)
#            
#            for i in range(self.Kgen_resblocks):
#                x = Conv(name = 'conv2_%s'%(i), x = x, filter_size = 3, in_filters = self.n_feats, out_filters = self.n_feats, strides = 1, padding = 'SAME')
#                x = tf.nn.relu(x)
#
#            #x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
#            x = Conv(name = 'conv_last', x = x, filter_size = 5, in_filters = self.n_feats, out_filters = 1, strides = 1, padding = 'SAME')
#            #x = tf.nn.sigmoid(x)
#            #x = x + _res
#            x = tf.clip_by_value(x, 0, 1.0)
#            x=x/(tf.math.reduce_sum(x))
#            return x
        
    def KGenerator(self, data, reuse=False, name='Kg_'):
        with tf.variable_scope(name,reuse=reuse):
            data_flatten = flatten(data)
            #tf_utils.print_activations(data_flatten)

            # from (N, 64) to (N, 4, 4, 128)
            h0_linear = tf_utils.linear(data_flatten, 4*4*128, name='h0_linear')

            h0_linear = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, 128])
            h0_linear = tf_utils.norm(h0_linear, _type='instance', name='h0_norm')
            h0_relu = tf.nn.relu(h0_linear, name='h0_relu')
            h0_reshape = tf.reshape(h0_relu, [tf.shape(h0_relu)[0], 4, 4, 128])

            # from (N, 4, 4, 128) to (N, 8, 8, 128)
            h1_deconv = tf_utils.deconv2d(h0_reshape, 128, k_h=5, k_w=5, name='h1_deconv2d',is_print=False)

            h1_deconv = tf_utils.norm(h1_deconv, _type='instance', name='h1_norm')
            h1_relu = tf.nn.relu(h1_deconv, name='h1_relu')

            # from (N, 8, 8, 128) to (N, 16, 16, 64)
            h2_deconv = tf_utils.deconv2d(h1_relu, 64, k_h=5, k_w=5, name='h2_deconv2d',is_print=False)

            h2_deconv = tf_utils.norm(h2_deconv, _type='instance', name='h2_norm')
            h2_relu = tf.nn.relu(h2_deconv, name='h2_relu')

            # from (N, 16, 16, 64) to (N, 32, 32, 32)
            h3_deconv = tf_utils.deconv2d(h2_relu, 32, k_h=5, k_w=5, name='h3_deconv2d',is_print=False)
            h3_deconv = tf_utils.norm(h3_deconv, _type='instance', name='h3_norm')
            h3_relu = tf.nn.relu(h3_deconv, name='h3_relu')
            
            # from (N, 32, 32, 32) to (N, 17,17, 1)
            h4 = tf_utils.conv2d(h3_relu,k=1,k_h=2, k_w=2, d_h=1, d_w=1, stddev=0.02, padding='VALID',name='h4_conv2d',is_print=False)
            x = tf.clip_by_value(h4, 0, 1.0)
            x=x/(tf.math.reduce_sum(x))
            
            return x
    
    def Kdiscriminator(self, x, reuse = False, name = 'Kd_'):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            x = Conv(name = 'conv1', x = x, filter_size = 4, in_filters = self.channel, out_filters = self.n_feats, strides = 2, padding = "SAME")
            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x = tf.nn.leaky_relu(x)
            
            prev = 1
            n = 1
            
            for i in range(self.Kdiscrim_blocks):
                prev = n
                n = min(2 ** (i+1), 8)
                x = Conv(name = 'conv%02d'%i, x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n, strides = 2, padding = "SAME")
                x = instance_norm(name = 'instance_norm%02d'%i, x = x, dim = self.n_feats * n)
                x = tf.nn.leaky_relu(x)
                
            prev = n
            n = min(2**self.discrim_blocks, 8)
            x = Conv(name = 'conv_d1', x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n, strides = 1, padding = "SAME")
            x = instance_norm(name = 'instance_norm_d1', x = x, dim = self.n_feats * n)
            x = tf.nn.leaky_relu(x)
            
            x = Conv(name = 'conv_d2', x = x, filter_size = 4, in_filters = self.n_feats * n, out_filters = 1, strides = 1, padding = "SAME")
            x = tf.nn.sigmoid(x)
            
            return x
        
    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.flags.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.gene_img - self.blur
        interpolates = self.blur + (alpha * differences)
        gradients = tf.gradients(self.Kdiscriminator(interpolates, is_reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty
        
    def build_graph(self):
        

        self.blur = tf.placeholder(name = "blur", shape = [None, None, None, self.channel], dtype = tf.float32)
        self.sharp = tf.placeholder(name = "sharp", shape = [None, None, None, self.channel], dtype = tf.float32)
        self.Knoise = tf.placeholder(name = "sharp", shape = [None,self.Kernel_size,self.Kernel_size, self.channel], dtype = tf.float32)
        blur = self.blur
        sharp = self.sharp
        Knoise = self.Knoise

        self.epoch = tf.placeholder(name = 'train_step', shape = None, dtype = tf.int32)
        
        blur = (2.0 * blur / 255.0) - 1.0
        sharp = (2.0 * sharp / 255.0) - 1.0
        
        
        self.gene_K = self.generator(Knoise, reuse = False)
        K=self.gene_K
        Kernel = tf.tile(tf.reshape(K,[self.Kernel_size,self.Kernel_size,1,1]),[1,1,3,1])
        sharp = tf.pad(sharp, [[0,0],[self.Kernel_size//2,self.Kernel_size//2],[self.Kernel_size//2,self.Kernel_size//2],[0,0]], mode = 'REFLECT')
        self.gene_img = tf.nn.depthwise_conv2d(sharp, Kernel, [1,1,1,1], padding = 'VALID')
        
        
        self.real_prob = self.discriminator(blur, reuse = False)
        self.fake_prob = self.discriminator(self.gene_img, reuse = True)
        
#        epsilon = tf.random_uniform(shape = [self.batch_size, 1, 1, 1], minval = 0.0, maxval = 1.0)
#        
#        interpolated_input = epsilon * label + (1 - epsilon) * self.gene_img
#        gradient = tf.gradients(self.discriminator(interpolated_input, reuse = True), [interpolated_input])[0]
#        GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis = [1, 2, 3])) - 1))
#        
        d_loss_real = - tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)
        self.gp_loss = self.gradient_penalty()
        if self.mode == 'train':
#            self.vgg_net = Vgg19(self.vgg_path)
#            self.vgg_net.build(tf.concat([label, self.gene_img], axis = 0))
#            self.content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]), axis = 3))
            
            self.D_loss = d_loss_real + d_loss_fake +self.gp_loss#+ 10.0 * GP_loss
            self.G_loss = - d_loss_fake #+ 100.0 * self.content_loss

#            t_vars = tf.trainable_variables()
#            G_vars = [var for var in t_vars if 'generator' in var.name]
#            D_vars = [var for var in t_vars if 'discriminator' in var.name]
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Kd_')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Kg_')

            lr = tf.minimum(self.learning_rate, tf.abs(2 * self.learning_rate - (self.learning_rate * tf.cast(self.epoch, tf.float32) / self.decay_step)))
            self.D_train = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.D_loss, var_list = D_vars)
            self.G_train = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.G_loss, var_list = G_vars)
            
            logging_D_loss = tf.summary.scalar(name = 'D_loss', tensor = self.D_loss)
            logging_G_loss = tf.summary.scalar(name = 'G_loss', tensor = self.G_loss)
        
#        self.PSNR = tf.reduce_mean(tf.image.psnr(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))
#        self.ssim = tf.reduce_mean(tf.image.ssim(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))
#        
#        logging_PSNR = tf.summary.scalar(name = 'PSNR', tensor = self.PSNR)
#        logging_ssim = tf.summary.scalar(name = 'ssim', tensor = self.ssim)
        
        self.output = (self.gene_img + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)

