import numpy as np

import tensorflow as tf

import importlib

import argparse

import keras

import foolbox

import datetime

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import keras.backend as K

from scipy.integrate import trapz

from tqdm import tqdm

from multivap import MultIVAP

from tabulate import tabulate

from whitebox import WhiteboxAttack

def normalize(arr):
    min_val, max_val = arr.min(), arr.max()
    arr -= min_val
    arr /= max_val - min_val
    return arr

def heatmap(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='gray')
    ax.set_xticks(np.arange(cm.shape[0]))
    ax.set_yticks(np.arange(cm.shape[1]))
    ax.set_xticklabels([i+1 for i in range(cm.shape[0])])
    ax.set_yticklabels([i+1 for i in range(cm.shape[1])])
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, int(cm[i, j]),
                           ha="center", va="center", color="red")
    fig.tight_layout()

def print_report(metrics, file):
    print(tabulate([
        ['Accuracy', metrics.acc],
        ['Efficiency', '{} <= {} <= {}'.format(metrics.eff_lower, metrics.eff, metrics.eff_upper)],
        ['Rejection', metrics.rej],
        ['TRR', metrics.trr],
        ['FRR', metrics.frr]
    ]), file=file)

def generate_advs(fmodel, attack, model, batch_size, x_trial, y_trial=None):
    if y_trial is None:
        y_trial = model.predict(x_trial, batch_size=batch_size)
    x_advs, y_advs = [], []
    for i, (x, y) in enumerate(zip(tqdm(x_trial, ascii=True), y_trial)):
        try:
            adv = foolbox.adversarial.Adversarial(
                    fmodel,
                    foolbox.criteria.Misclassification(),
                    x,
                    y.argmax(),
                    foolbox.distances.Linfinity)
            current_result = attack(adv)
            if current_result is not None:
                x_advs.append(current_result)
                y_advs.append(y)
            else:
                x_advs.append(x)
                y_advs.append(y)
        except ValueError:
            x_advs.append(x)
            y_advs.append(y)
    x_advs = np.array(x_advs)
    y_advs = np.array(y_advs)
    
    return x_advs, y_advs

def gen_curves(x_samples, y_samples, beta):
    epses = np.linspace(0, 1, 100)
    accs, eff_lowers, effs, eff_uppers, rejs = [], [], [], [], []
    for eps in tqdm(epses, ascii=True):
        cm, metrics = multivap.evaluate(x_samples, y_samples, eps)

        accs.append(metrics.acc)
        eff_lowers.append(metrics.eff_lower)
        effs.append(metrics.eff)
        eff_uppers.append(metrics.eff_upper)
        rejs.append(metrics.rej)
    
    fig, axes = plt.subplots(nrows=3, sharex=True)
    axes[0].set_ylabel('accuracy')
    axes[0].plot(epses, accs)
    axes[0].axvline(x=beta, color='black', ls=':')
    axes[1].set_ylabel('efficiency')
    axes[1].plot(epses, effs)
    axes[1].axvline(x=beta, color='black', ls=':')
    axes[1].fill_between(epses, np.array(eff_lowers), np.array(eff_uppers), alpha=.5)
    axes[2].set_ylabel('rejection')
    axes[2].plot(epses, rejs)
    axes[2].axvline(x=beta, color='black', ls=':')
    axes[2].set_xlabel('significance level')

def roc_curves(x_samples, y_samples, beta):
    epses = np.linspace(0, 1, 100)
    trrs, frrs = [], []
    for eps in tqdm(epses, ascii=True):
        cm, metrics = multivap.evaluate(x_samples, y_samples, eps)

        trrs.append(metrics.trr)
        frrs.append(metrics.frr)
    
    roc = list(zip(frrs, trrs))
    sorted(roc, key=lambda x: x[0])
    roc = np.array(roc)
    auc = trapz(roc[:,1], roc[:,0])

    cm, metrics = multivap.evaluate(x_samples, y_samples, beta)

    plt.title('AUC = {}'.format(auc))
    plt.plot(roc[:,0], roc[:,1], c='red', label='ROC')
    plt.plot([0, 1], [0, 1], ls='--', c='orange', label='chance')
    plt.axvline(metrics.frr, c='black', ls=':', label='threshold')
    plt.xlabel('FRR')
    plt.ylabel('TRR')
    plt.legend()

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate the MultIVAP.')
    parser.add_argument('task', metavar='T', type=str, help='task to evaluate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--frac', type=float, default=.2, help='GPU memory fraction')
    parser.add_argument('--eta', type=float, default=.1, help='perturbation budget for white-box attack')

    args = parser.parse_args()

    # limit TF memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.frac
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # load task
    module = importlib.import_module(args.task)

    # load data
    x_train, y_train, x_test, y_test = module.load_datasets()

    # normalize the data set
    x_train = normalize(x_train.astype(np.float64))
    x_test = normalize(x_test.astype(np.float64))

    # splits for proper training, proper testing, calibration and validation sets
    idx = int(.8*x_train.shape[0])
    x_proper_train, y_proper_train = x_train[:idx], y_train[:idx]
    x_calib, y_calib = x_train[idx:], y_train[idx:]

    idx = int(.8*x_test.shape[0])
    x_proper_test, y_proper_test = x_test[:idx], y_test[:idx]
    x_valid, y_valid = x_test[idx:], y_test[idx:]

    # reporting
    with open('reports/{}.md'.format(args.task), 'w') as report:
        print('# Task report: {}\n'.format(args.task), file=report)

        print('## Data set summary\n', file=report)
        print('Training samples: {}\n'.format(x_proper_train.shape), file=report)
        print('Calibration samples: {}\n'.format(x_calib.shape), file=report)
        print('Test samples: {}\n'.format(x_test.shape), file=report)
        print('Validation samples: {}\n'.format(x_valid.shape), file=report)

        # fit the model
        model = module.create_model()
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=module.get_optimizer(),
                        metrics=['accuracy'])
        model.fit(x_proper_train, y_proper_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=1,
                validation_data=(x_valid, y_valid))
    
        # report overall accuracy
        print('## Baseline model\n', file=report)
        start = datetime.datetime.now()
        acc = model.evaluate(x_proper_test, y_proper_test, batch_size=args.batch_size)[1]
        end = datetime.datetime.now()
        avg_diff = (end - start).total_seconds() / x_proper_test.shape[0]
        print('Raw baseline accuracy: {}\n'.format(acc), file=report)
        print('Average time: {}\n'.format(avg_diff), file=report)
    
        # calibrate the MultIVAP
        print('## MultIVAP\n', file=report)
        multivap = MultIVAP(model, x_calib, y_calib, y_test.shape[1])

        # tune the MultIVAP
        best_beta, _ = multivap.tune(x_valid)
        print('Significance level: {}\n'.format(best_beta), file=report)

        # test the MultIVAP on regular test data
        cm, metrics = multivap.evaluate(x_proper_test, y_proper_test, best_beta)
        print('Proper test set:\n', file=report)
        print_report(metrics, report)

        heatmap(cm)
        plt.savefig('plots/{}_heatmap_clean.pdf'.format(args.task))
        plt.clf()

        gen_curves(x_proper_test, y_proper_test, best_beta)
        plt.savefig('plots/{}_curves_clean.pdf'.format(args.task))
        plt.clf()

        roc_curves(x_proper_test, y_proper_test, best_beta)
        plt.savefig('plots/{}_roc.pdf'.format(args.task))
        plt.clf()

        # time the MultIVAP
        start = datetime.datetime.now()
        multivap.predict(x_proper_test, best_beta)
        end = datetime.datetime.now()
        avg_diff = (end - start).total_seconds() / x_proper_test.shape[0]
        print('Average time: {}\n'.format(avg_diff), file=report)

        # compare with baseline
        accepted = multivap.predict(x_proper_test, best_beta).sum(axis=1).nonzero()[0]
        x_accepted, y_accepted = x_proper_test[accepted], y_proper_test[accepted]
        acc = model.evaluate(x_accepted, y_accepted, batch_size=args.batch_size)[1]
        print('Corrected baseline accuracy: {}\n'.format(acc), file=report)

        # generate white-box adversarials
        print('Running white-box attack...')
        whitebox = WhiteboxAttack(multivap, model, x_calib, y_calib, batch_size=args.batch_size)
        x_advs, flags = whitebox.attack(x_proper_test, y_proper_test, eta=args.eta, beta=best_beta, its=100)

        print('### White-box attack\n', file=report)
        print('Perturbation budget: {}\n'.format(args.eta), file=report)
        print('Success rate: {}\n'.format(flags.mean()), file=report)

        cm, metrics = multivap.evaluate(x_advs[flags], y_proper_test[flags], best_beta)
        print('Metrics:\n', file=report)
        print_report(metrics, report)

        # compare with baseline
        acc = model.evaluate(x_advs[flags], y_proper_test[flags], batch_size=args.batch_size)[1]
        print('Baseline accuracy: {}\n'.format(acc), file=report)

        # generate adversarials using each attack
        attacks = [
            foolbox.attacks.DeepFoolAttack,
            foolbox.attacks.RandomPGD,
            foolbox.attacks.FGSM,
            foolbox.attacks.SinglePixelAttack,
            foolbox.attacks.LocalSearchAttack
        ]
        fmodel = foolbox.models.KerasModel(model, (0, 1))
        for attack_class in attacks:
            attack = attack_class(fmodel)

            print('Generating adversarials using {}...'.format(attack.__class__.__name__))
            x_advs, y_advs = generate_advs(fmodel, attack, model, args.batch_size, x_proper_test)

            # test the MultIVAP on these adversarials
            print('### {}\n'.format(attack.__class__.__name__), file=report)
            cm, metrics = multivap.evaluate(x_advs, y_advs, best_beta)
            print_report(metrics, report)

            # compare with baseline
            acc = model.evaluate(x_advs, y_advs, batch_size=args.batch_size)[1]
            print('Raw baseline accuracy: {}\n'.format(acc), file=report)

            accepted = multivap.predict(x_proper_test, best_beta).sum(axis=1).nonzero()[0]
            x_accepted, y_accepted = x_advs[accepted], y_advs[accepted]
            acc = model.evaluate(x_accepted, y_accepted, batch_size=args.batch_size)[1]
            print('Corrected baseline accuracy: {}\n'.format(acc), file=report)
