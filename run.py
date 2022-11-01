from easydict import EasyDict
import pickle
from trainer import trainer
import matplotlib.pyplot as plt

def single_exp(config):
    config = EasyDict(config)
    print(config)

    data = pickle.load(open('data/{}/train_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']

    exp_trainner = trainer(config, data)
    exp_trainner.train()

    plt.plot(exp_trainner.train_loss_list, label='train')
    plt.plot(exp_trainner.test_loss_list, label='test')
    plt.legend()
    plt.title('losses')
    plt.savefig('figures/{}/{}/losses_fold{}.png'.format(config.data_name, config.model_name, config.fold))
    plt.show()

    plt.figure()
    plt.plot(exp_trainner.test_roc_auc_list, label = 'test roc auc')
    plt.plot(exp_trainner.test_pr_auc_list, label = 'test pr auc')
    plt.legend()
    plt.title('evaluation')
    plt.savefig('figures/{}/{}/evaluation_fold{}.png'.format(config.data_name, config.model_name, config.fold))
    plt.show()


def ednet():
    config = {
        "data_name": 'ednet',
        "model_name": 'TAMKOT',
        "mode": 'test',
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "min_seq_len": 2,
        "max_seq_len": 50,  # the max step of TAMKOT model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 200,
        "validation_split": 0.2,

        "embedding_size_q": 64,
        "embedding_size_a": 64,
        "embedding_size_l": 32,
        "hidden_size": 16,

        "init_std": 0.2,
        "max_grad_norm": 10,

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
        "mode": 'test',
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max step of TAMKOT model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 200,
        "validation_split": 0.2,

        "embedding_size_q": 32,
        "embedding_size_a": 32,
        "embedding_size_l": 8,
        "hidden_size": 32,

        "init_std": 0.2,
        "max_grad_norm": 50,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.05,
    }
    single_exp(config)


def morf():
    config = {
        "data_name": 'MORF686',
        "model_name": 'TAMKOT',
        "mode": 'test',
        "fold": 4,
        "metric": 'rmse',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max step of TAMOKOT model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 200,
        "validation_split": 0.2,

        "embedding_size_q": 64,
        "embedding_size_a": 64,
        "embedding_size_l": 8,
        "hidden_size": 16,

        "init_std": 0.2,
        "max_grad_norm": 50,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.5,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    single_exp(config)



if __name__== '__main__':
    # ednet()
    # Junyi2063()
    morf()
