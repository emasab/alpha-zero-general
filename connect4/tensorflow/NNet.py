import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet
from tensorflow.python.saved_model import signature_constants

import tensorflow as tf
from .Connect4NNet import Connect4NNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})


## Code based on othello.NNetWrapper with minimal changes.

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.sess = tf.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                input_dict = {self.nnet.input_boards: boards, self.nnet.target_pis: pis, self.nnet.target_vs: vs, self.nnet.dropout: args.dropout, self.nnet.isTraining: True}

                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict)
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples) / args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        prob, v = self.sess.run([self.nnet.prob, self.nnet.v], feed_dict={self.nnet.input_boards: board, self.nnet.dropout: 0, self.nnet.isTraining: False})

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return prob[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def _write_saved_model(self, saved_model_path,
                        inputs,
                        outputs):
        """Writes SavedModel to disk.
        Args:
            saved_model_path: Path to write SavedModel.
            trained_checkpoint_prefix: path to trained_checkpoint_prefix.
            inputs: The input image tensor to use for detection.
            outputs: A tensor dictionary containing the outputs of a DetectionModel.
        """
        with self.sess as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

            tensor_info_inputs = {}
            tensor_info_outputs = {}
            for k, v in inputs.items():
                tensor_info_inputs[k] = tf.saved_model.utils.build_tensor_info(v)
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )

        builder.save()
        

    def save_simple_model(self, out="out"):
        self._write_saved_model(out, {"board": self.nnet.input_boards, "dropout": self.nnet.dropout, "isTraining": self.nnet.isTraining},{"prob": self.nnet.prob, "v": self.nnet.v})


    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + '.meta'):
            raise("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)
