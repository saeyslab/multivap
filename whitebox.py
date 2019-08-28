import keras
import keras.backend as K

import numpy as np

import tensorflow as tf

from keras.models import Model

from tqdm import trange

class WhiteboxAttack:
    def __init__(self, multivap, model, x_calib, y_calib, batch_size=128, lr=1e-4):
        # store parameters
        self.multivap = multivap
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.batch_size = batch_size
        
        # get calibration scores
        self.logits = Model(inputs=model.input, outputs=model.layers[-2].output)
        self.calib_scores = self.logits.predict(x_calib)
        
        # construct optimization
        sess = K.get_session()
        with tf.variable_scope('whitebox', reuse=tf.AUTO_REUSE):
            self.eta = tf.placeholder(tf.float32)
            self.lams = tf.placeholder(tf.float32)
            self.x_origs = tf.placeholder(tf.float32)
            self.x_tildes = tf.get_variable('x_tilde', shape=[self.batch_size, *x_calib.shape[1:]])
            self.target_scores = tf.placeholder(tf.float32)
            
            logits_tensor = self.logits(self.x_tildes)
            loss = tf.reduce_mean(
                tf.reduce_max(abs(self.x_tildes - self.x_origs), axis=[1, 2, 3]) \
                + self.lams * tf.reduce_max(abs(logits_tensor - self.target_scores), axis=1))
            self.opt_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[self.x_tildes])

            init_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='whitebox')
            self.init_op = [tf.variables_initializer(init_vars),
                           tf.assign(self.x_tildes, self.x_origs)]
            self.clip_op = tf.assign(self.x_tildes, tf.clip_by_value(self.x_tildes, 0, 1))

    def batch_attack(self, x_origs, y_origs, eta, beta, its=10, tol=1e-2):
        # sanity check
        assert x_origs.shape[0] == self.batch_size, 'Batches must be {} samples each'.format(self.batch_size)
        
        # get original scores
        orig_scores = self.logits.predict(x_origs)
        
        # find appropriate target vectors
        target_scores = np.copy(orig_scores)
        for idx, (x_orig, y_orig, orig_score) in enumerate(zip(x_origs, y_origs, orig_scores)):
            candidates = [(x, y, s) for x, y, s in zip(self.x_calib, self.y_calib, self.calib_scores) if y.argmax() != y_orig.argmax()]
            dists = -np.array([abs(orig_score - s).max() for x, y, s in candidates])
            zs = np.exp(dists - dists.max())
            probs = zs / np.sum(zs)

            candidate_scores = np.array([s for x, y, s in candidates])
            candidate_idx = np.random.choice(list(range(len(candidates))), p=probs)
            target_scores[idx] = np.copy(candidate_scores[candidate_idx])

        # optimize lambdas
        lowers, uppers = np.zeros(self.batch_size), np.ones(self.batch_size)
        x_sols = np.copy(x_origs)
        flags = np.zeros(self.batch_size).astype(np.bool)
        while (uppers - lowers).max() > tol:
            lams = lowers + (uppers - lowers)/2
            mask = np.zeros(self.batch_size).astype(np.bool)

            # start optimization
            sess = K.get_session()
            x_inits = np.clip(x_origs, 0, 1)
            sess.run(self.init_op, feed_dict={self.x_origs: x_inits})

            # optimization loop
            for it in range(its):
                # optimization step
                sess.run(self.opt_op, feed_dict={self.lams: lams,
                                            self.x_origs: x_origs,
                                            self.target_scores: target_scores,
                                            self.eta: eta})
                sess.run(self.clip_op)

                # check intermediate solutions
                x_tildes_raw = sess.run(self.x_tildes)
                y_tildes = self.multivap.predict(x_tildes_raw, beta)
                
                for idx in range(self.batch_size):
                    y_true = y_origs[idx].argmax()
                    if y_tildes[idx].sum() > 0 and \
                            abs(x_tildes_raw[idx] - x_origs[idx]).max() <= eta and \
                            y_tildes[idx][y_true] == 0:
                        x_sols[idx] = np.copy(x_tildes_raw[idx])
                        mask[idx] = True
                        flags[idx] = True

            # update lambdas
            uppers[mask] = lams[mask]
            lowers[np.logical_not(mask)] = lams[np.logical_not(mask)]

        return x_sols, flags
    
    def attack(self, x_origs, y_origs, eta, beta, its=10, tol=1e-2, verbose=True):
        num_batches = x_origs.shape[0] // self.batch_size
        x_advs = np.zeros(x_origs.shape)
        flags = np.zeros(x_origs.shape[0]).astype(np.bool)
        t = range(num_batches) if not verbose else trange(num_batches)
        for idx in t:
            start = idx * self.batch_size
            end = (idx+1) * self.batch_size
            x_advs[start:end], flags[start:end] = self.batch_attack(x_origs[start:end], y_origs[start:end], eta, beta, its, tol)
        return x_advs, flags
