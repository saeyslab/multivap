import numpy as np

import keras.backend as K

import io

import cvxopt
from cvxopt import solvers, matrix, spdiag, log
from cvxopt.glpk import ilp

from VennABERS import ScoresToMultiProbs, computeF, getFVal, prepareData

from collections import namedtuple

class MultIVAP:
    def __init__(self, model, x_calib, y_calib, num_classes):
        self.ivaps = []
        self.num_classes = num_classes
        self.model = model
        for i in range(num_classes):
            self.ivaps.append(IVAP(model, x_calib, y_calib, i, num_classes))
    
    def _score(self, samples, batch_size=128):
        scores = np.zeros((samples.shape[0], self.num_classes, 2))
        for i, ivap in enumerate(self.ivaps):
            scores[:, i, 0], scores[:, i, 1] = ivap.batch_predictions(samples, batch_size)
        return scores
    
    def predict(self, samples, beta, batch_size=128, scores=None, verbose=False):
        cvxopt.solvers.options['show_progress'] = verbose
        cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON' if verbose else 'GLP_MSG_OFF'
        if scores is None:
            scores = self._score(samples, batch_size)
        
        probs = []
        for score in scores:
            p0s, p1s = score[:, 0], score[:, 1]
            K = self.num_classes
            if all(p1s < beta) or all(p0s < beta):
                probs.append(np.zeros(K))
            else:
                c = np.ones(K+1)
                c = -matrix(c)

                G = np.zeros([K + 1, K + 1])
                G[0, :K] = 1 - p0s
                G[1:, -1] = np.ones(K)
                for i in range(K):
                    G[i + 1, i] = 1 - p1s[i]
                G = matrix(G)

                h = matrix(np.concatenate((
                    np.array([1 - beta]),
                    np.ones(K)
                )))

                B = set(range(K))

                _, x = ilp(c=c, G=G, h=h, B=B, I=set())
                if x is None:
                    probs.append(np.zeros(K))
                else:
                    probs.append(np.array(x)[:-1].reshape(K))
        return np.array(probs)
    
    def tune(self, samples, batch_size=128, tol=1e-7):
        scores = self._score(samples, batch_size)
        beta, eff = 0, 0
        lower, upper = 0, 1
        while upper - lower > tol:
            beta = lower + (upper-lower)/2
            probs = self.predict(samples, beta, batch_size, scores)
            eff = probs.sum(axis=1).mean()
            if eff < 1:
                upper = beta
            else:
                lower = beta
        return beta, eff

    def evaluate(self, samples, labels, beta, batch_size=128):
        probs = self.predict(samples, beta, batch_size)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                confusion_matrix[i, j] = np.sum([y_true == i and y_pred[j] == 1
                                                 for y_true, y_pred in zip(labels.argmax(axis=1),
                                                                            probs)
                                                if y_pred.sum() > 0])
        
        Metrics = namedtuple('Metrics', ['eff', 'eff_lower', 'eff_upper', 'acc', 'rej', 'trr', 'frr'])
        sizes = probs.sum(axis=1)

        try:
            eff = sizes.mean()
            eff_lower = np.percentile(sizes, 5)
            eff_upper = np.percentile(sizes, 95)
        except:
            eff, eff_lower, eff_upper = np.nan, np.nan, np.nan
        rejected = (sizes == 0)
        accepted = np.logical_not(rejected)
        rej = np.mean(rejected)
        acc = np.diag(confusion_matrix).sum() / accepted.sum() if accepted.sum() > 0 else np.nan

        y_model = self.model.predict(samples, batch_size=batch_size).argmax(axis=1)
        trs = np.sum([y_true != y_pred for y_true, y_pred in zip(labels[rejected].argmax(axis=1), y_model[rejected])])
        frs = np.sum([y_true == y_pred for y_true, y_pred in zip(labels[rejected].argmax(axis=1), y_model[rejected])])
        tas = np.sum([y_true == y_pred for y_true, y_pred in zip(labels[accepted].argmax(axis=1), y_model[accepted])])
        fas = np.sum([y_true != y_pred for y_true, y_pred in zip(labels[accepted].argmax(axis=1), y_model[accepted])])

        trr = trs / (trs + fas)
        frr = frs / (frs + tas)

        return confusion_matrix, Metrics(eff, eff_lower, eff_upper, acc, rej, trr, frr)

class IVAP:
    def __init__(self, model, x_calib, y_calib, class_idx=0, num_classes=2):
        self.model = model
        self.cidx = class_idx
        self.num_classes = num_classes
        self.get_logits = K.function([self.model.layers[0].input],
                                  [self.model.layers[-2].output])

        # prepare isotonic regression
        self.x_calib, self.y_calib = x_calib, y_calib
        self.calib_points = [(score, label) for score, label in zip(self._score(x_calib)[:, self.cidx], np.argmax(y_calib, axis=1) == self.cidx)]
        yPrime, yCsd, xPrime, self.ptsUnique = prepareData(self.calib_points)
        self.F0, self.F1 = computeF(xPrime, yCsd)
    
    def _score(self, images, batch_size=128):
        scores = np.zeros((images.shape[0], self.num_classes))
        num_batches = images.shape[0] // batch_size
        for i in range(num_batches):
            start = i*batch_size
            end = (i+1)*batch_size
            batch = images[start:end]
            scores[start:end,:] = self.get_logits([batch])[0]
        return scores
    
    def batch_predictions(self, images, batch_size=128):
        logits = np.zeros((images.shape[0], self.num_classes))
        num_batches = images.shape[0] // batch_size
        for i in range(num_batches):
            start = i*batch_size
            end = (i+1)*batch_size
            batch = images[start:end]
            logits[start:end,:] = self._score(batch)

        p0s, p1s = getFVal(self.F0, self.F1, self.ptsUnique, logits[:, self.cidx])
        return p0s, p1s
