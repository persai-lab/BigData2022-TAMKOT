from easydict import EasyDict
import pickle
from trainer import trainer
import matplotlib.pyplot as plt


def single_exp(config):
    config = EasyDict(config)
    print(config)

    data = pickle.load(open('data/{}/train_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']  # number of assessed materials
    config.num_nongradable_items = data['num_items_L']  # number of non-assessed materials
    config.num_users = data['num_users'] # number of student

    exp_trainner = trainer(config, data)
    exp_trainner.train()

    # plot and save losses and evaluation metrics
    plt.plot(exp_trainner.train_loss_list, label='train')
    plt.plot(exp_trainner.test_loss_list, label='test')
    plt.legend()
    plt.title('losses')
    plt.savefig('figures/{}/{}/losses_fold{}.png'.format(config.data_name, config.model_name, config.fold))
    plt.show()

    plt.figure()
    plt.plot(exp_trainner.test_roc_auc_list, label='test roc auc')
    plt.plot(exp_trainner.test_pr_auc_list, label='test pr auc')
    plt.legend()
    plt.title('evaluation')
    plt.savefig('figures/{}/{}/evaluation_fold{}.png'.format(config.data_name, config.model_name, config.fold))
    plt.show()


def ednet():
    config = {
        "data_name": 'ednet',
        "model_name": 'TAMKOT',
        "mode": 'test',  # 'train' or 'test', 'train': for each fold's train set, split a validation set as test data
        # for tunning hyperparameters. 'test': test set as test data for final model evaluation
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "max_seq_len": 50,  # the max sequence length of TAMKOT model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 100,  # the max training epoch
        "validation_split": 0.2,  # the ratio of splitting validation set from training set, used if "mode": 'train'

        "embedding_size_q": 64,  # assessed materials embedding size
        "embedding_size_a": 64,  # student performance embedding size
        "embedding_size_l": 32,  # non-assessed materials embedding size
        "hidden_size": 16,  # hidden state size

        "init_std": 0.2,
        "max_grad_norm": 5,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    single_exp(config)


def Junyi2063():
    config = {
        "data_name": 'Junyi2063',
        "model_name": 'TAMKOT',
        "mode": 'test',   # 'train' or 'test', 'train': for each fold's train set, split a validation set as test data
        # for tunning hyperparameters. 'test': test set as test data for final model evaluation
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "max_seq_len": 100,  # the max sequence length of TAMKOT model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 100,  # the max training epoch
        "validation_split": 0.2,  # the ratio of splitting validation set from training set, used if "mode": 'train'

        "embedding_size_q": 32,  # assessed materials embedding size
        "embedding_size_a": 32,  # student performance embedding size
        "embedding_size_l": 32,  # non-assessed materials embedding size
        "hidden_size": 32,  # hidden state size

        "init_std": 0.2,
        "max_grad_norm": 50,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.05,
    }
    single_exp(config)


if __name__ == '__main__':
    # ednet()
    Junyi2063()
