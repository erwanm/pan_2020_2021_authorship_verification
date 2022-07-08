# -*- coding: utf-8 -*-
from adhominem_o2d2 import AdHominem_O2D2
import pickle
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description='AdHominem for PAN 2020 and 2021')
    #
    parser.add_argument('-thr_0', default=0.3, type=float)  # lower threshold for O2D2
    parser.add_argument('-thr_1', default=0.7, type=float)  # upper threshold for O2D2
    parser.add_argument('-epoch_trained', default=32, type=int)  # best epoch of trained AdHominem model
    #
    parser.add_argument('-lr_start', default=0.001, type=float)  # initial learning rate
    parser.add_argument('-lr_end', default=0.0005, type=float)  # lower bound for learning rate
    parser.add_argument('-lr_epoch', default=100, type=float)  # epoch, when achieving the lower bound
    #
    parser.add_argument('-epochs', default=10, type=int)  # total number of epochs
    parser.add_argument('-batch_size', default=30, type=int)  # batch size for training
    parser.add_argument('-batch_size_val', default=30, type=int)  # batch size for evaluation
    #
    parser.add_argument('-retrain_chr_emb', default=False, type=bool)  # retrain certain layers
    parser.add_argument('-retrain_wrd_emb', default=False, type=bool)
    parser.add_argument('-retrain_cnn', default=False, type=bool)
    parser.add_argument('-retrain_bilstm', default=False, type=bool)
    parser.add_argument('-retrain_dml', default=False, type=bool)
    parser.add_argument('-retrain_loss_dml', default=False, type=bool)
    parser.add_argument('-retrain_bfs', default=False, type=bool)
    parser.add_argument('-retrain_ual', default=False, type=bool)
    #
    parser.add_argument('-keep_prob_cnn', default=1.0, type=float)  # apply dropout when computing LEVs
    parser.add_argument('-keep_prob_lstm', default=1.0, type=float)
    parser.add_argument('-keep_prob_att', default=1.0, type=float)
    parser.add_argument('-keep_prob_metric', default=1.0, type=float)
    parser.add_argument('-keep_prob_bfs', default=1.0, type=float)
    parser.add_argument('-keep_prob_ual', default=1.0, type=float)
    #
    parser.add_argument('-rate_o2d2', default=0.25, type=float) # O2D2 dropout rate
    #
    hyper_parameters_new = vars(parser.parse_args())

    # create folder for results
    dir_results = os.path.join('..', 'results_o2d2')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # load trained model and hyper-parameters
    with open(os.path.join("..",
                           "results_adhominem",
                           'weights_adhominem',
                           "weights_" + str(hyper_parameters_new["epoch_trained"]),
                           ), 'rb') as f:
        parameters = pickle.load(f)

    # overwrite old variables
    for hyper_parameter in hyper_parameters_new:
        parameters["hyper_parameters"][hyper_parameter] = hyper_parameters_new[hyper_parameter]

    # load validation set
    with open(os.path.join('..', 'data_preprocessed', "pairs_val"), 'rb') as f:
        docs_L, docs_R, labels, _ = pickle.load(f)
    val_set = (docs_L, docs_R, labels)
    parameters["hyper_parameters"]['N_val'] = len(labels)

    # file to store results epoch-wise
    file_results = os.path.join(dir_results, 'results.txt')
    # temporary file to store results batch-wise
    file_tmp = os.path.join(dir_results, 'tmp.txt')

    # delete already existing files
    if os.path.isfile(file_results):
        os.remove(file_results)
    if os.path.isfile(file_tmp):
        os.remove(file_tmp)

    # write hyper-parameters setup into file (results.txt)
    open(file_results, 'a').write('\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\nPARAMETER SETUP:\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\n'
                                  )
    for hp in sorted(parameters["hyper_parameters"].keys()):
        if hp in ['V_c', 'V_w']:
            open(file_results, 'a').write('num ' + hp + ': ' + str(len(parameters["hyper_parameters"][hp])) + '\n')
        else:
            open(file_results, 'a').write(hp + ': ' + str(parameters["hyper_parameters"][hp]) + '\n')

    # load neural network model
    adhominem_o2d2 = AdHominem_O2D2(hyper_parameters=parameters['hyper_parameters'],
                                    theta_init=parameters['theta'],
                                    theta_E_init=parameters['theta_E'],
                                    )
    # start training
    adhominem_o2d2.train_model(val_set, file_results, file_tmp, dir_results)
    # close session
    adhominem_o2d2.sess.close()


if __name__ == '__main__':
    main()




# epoch of best run (AdHominem-O2D2 model)
EPOCH = 8
# define batch size
BATCH_SIZE = 4





dir_results = os.path.join("..", "results_o2d2")
#dir_results = os.path.join("..", "pretrained_models", "results_o2d2")

# load dev set
#with open(os.path.join(dir_data, "pairs_val"), 'rb') as f:
#    docs_L, docs_R, labels, _ = pickle.load(f)
#labels = np.array(labels)

# docs_L, docs_R, labels = docs_L[100:201], docs_R[100:201], labels[100:201]

dev_set = val_set 



# inference
print("start inference direct...")
pred_dml, pred_bfs, pred_ual, pred_o2d2, n_miss, conf_matrix, lev_L, lev_R, att_w_L, att_w_R, att_s_L, att_s_R \
    = adhominem_o2d2.evaluate(docs_L, docs_R, batch_size=BATCH_SIZE)

f = open('predict_debug_direct.tsv', 'w')
for i,v in enumerate(pred_ual):
    f.write('%d\t%f\t%f\n' % (i,v,pred_o2d2[i]))
f.close()

# compute confidence scores (p if p >= 0.5, otherwise 1-p)
conf_dml, labels_dml = adhominem.compute_confidence(pred_dml)
conf_bfs, labels_bfs = adhominem.compute_confidence(pred_bfs)
conf_ual, labels_ual = adhominem.compute_confidence(pred_ual)
conf_o2d2, labels_o2d2 = adhominem.compute_confidence(pred_o2d2)

# store data
print("store results (predictions, levs)...")
with open(os.path.join(dir_results, "results_att_lev_pred"), 'wb') as f:
    pickle.dump((pred_dml, pred_bfs, pred_ual, pred_o2d2,
                 n_miss, conf_matrix, lev_L, lev_R,
                 conf_dml, conf_bfs, conf_ual, conf_o2d2,
                 labels_dml, labels_bfs, labels_ual, labels_o2d2,
                 att_w_L, att_w_R, att_s_L, att_s_R,
                 ), f)

# print results
print("evaluate...")

print("PAN (dml): ", evaluate_all(pred_y=pred_dml, true_y=labels))
print("PAN (bfs)", evaluate_all(pred_y=pred_bfs, true_y=labels))
print("PAN (ual)", evaluate_all(pred_y=pred_ual, true_y=labels))
print("PAN (o2d2)", evaluate_all(pred_y=pred_o2d2, true_y=labels))

print("# non-responses:", n_miss)
print("Calibration (dml)", compute_calibration(true_labels=labels, pred_labels=labels_dml, confidences=conf_dml))
print("Calibration (bfs)", compute_calibration(true_labels=labels, pred_labels=labels_bfs, confidences=conf_bfs))
print("Calibration (ual)", compute_calibration(true_labels=labels, pred_labels=labels_ual, confidences=conf_ual))
print("Calibration (o2d2)", compute_calibration(true_labels=labels, pred_labels=labels_o2d2, confidences=conf_o2d2))


# load model
print("load trained model and hyper-parameters...")
path = os.path.join(dir_results, "weights_o2d2", "weights_" + str(EPOCH))
with open(path, 'rb') as f:
    parameters = pickle.load(f)

# build Tensorflow graph with trained weights
print("build tensorflow graph...")
adhominem = AdHominem_O2D2(hyper_parameters=parameters['hyper_parameters'],
                           theta_init=parameters['theta'],
                           theta_E_init=parameters['theta_E'],
                           )

# inference
print("start inference loaded...")
pred_dml, pred_bfs, pred_ual, pred_o2d2, n_miss, conf_matrix, lev_L, lev_R, att_w_L, att_w_R, att_s_L, att_s_R \
    = adhominem.evaluate(docs_L, docs_R, batch_size=BATCH_SIZE)

f = open('predict_debug_loaded.tsv', 'w')
for i,v in enumerate(pred_ual):
    f.write('%d\t%f\t%f\n' % (i,v,pred_o2d2[i]))
f.close()

# compute confidence scores (p if p >= 0.5, otherwise 1-p)
conf_dml, labels_dml = adhominem.compute_confidence(pred_dml)
conf_bfs, labels_bfs = adhominem.compute_confidence(pred_bfs)
conf_ual, labels_ual = adhominem.compute_confidence(pred_ual)
conf_o2d2, labels_o2d2 = adhominem.compute_confidence(pred_o2d2)

# store data
print("store results (predictions, levs)...")
with open(os.path.join(dir_results, "results_att_lev_pred"), 'wb') as f:
    pickle.dump((pred_dml, pred_bfs, pred_ual, pred_o2d2,
                 n_miss, conf_matrix, lev_L, lev_R,
                 conf_dml, conf_bfs, conf_ual, conf_o2d2,
                 labels_dml, labels_bfs, labels_ual, labels_o2d2,
                 att_w_L, att_w_R, att_s_L, att_s_R,
                 ), f)

# print results
print("evaluate...")

print("PAN (dml): ", evaluate_all(pred_y=pred_dml, true_y=labels))
print("PAN (bfs)", evaluate_all(pred_y=pred_bfs, true_y=labels))
print("PAN (ual)", evaluate_all(pred_y=pred_ual, true_y=labels))
print("PAN (o2d2)", evaluate_all(pred_y=pred_o2d2, true_y=labels))

print("# non-responses:", n_miss)
print("Calibration (dml)", compute_calibration(true_labels=labels, pred_labels=labels_dml, confidences=conf_dml))
print("Calibration (bfs)", compute_calibration(true_labels=labels, pred_labels=labels_bfs, confidences=conf_bfs))
print("Calibration (ual)", compute_calibration(true_labels=labels, pred_labels=labels_ual, confidences=conf_ual))
print("Calibration (o2d2)", compute_calibration(true_labels=labels, pred_labels=labels_o2d2, confidences=conf_o2d2))

print("finished inference...")

# close session
adhominem.sess.close()

