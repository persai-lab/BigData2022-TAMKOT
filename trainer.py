import logging
import numpy as np
from sklearn import metrics
from torch.backends import cudnn
import torch
from torch import nn
import warnings
from model.TAMKOT import TAMKOT
from dataloader import TAMKOT_DataLoader

warnings.filterwarnings("ignore")
cudnn.benchmark = True


class trainer(object):
    def __init__(self, config, data):
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.metric = config.metric

        self.mode = config.mode
        self.manual_seed = config.seed

        # initialize the data_loader, which include preprocessing the data
        self.data_loader = TAMKOT_DataLoader(config, data)


        self.current_epoch = 1
        self.current_iteration = 1

        # if student performance is binary correctness, AUC the larger the better.
        # if student performance is numerical grade, rmse the small the better
        if self.metric == "rmse":
            self.best_val_perf = 1.
        elif self.metric == "auc":
            self.best_val_perf = 0.

        # create empty list to store losses and testing evaluation metrics of each epoch
        self.train_loss_list = []
        self.train_loss_l_list = []
        self.test_loss_list = []
        self.test_roc_auc_list = []
        self.test_pr_auc_list = []
        self.test_rmse_list = []

        # build models
        self.model = TAMKOT(config)

        # define criterion
        self.criterion = nn.BCELoss(reduction='sum')
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.config.learning_rate,
                                       momentum=self.config.momentum,
                                       weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        betas=(config.beta1, config.beta2),
                                        eps=self.config.epsilon,
                                        weight_decay=self.config.weight_decay)



        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")


    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            print("="*50 + "Epoch {}".format(epoch) + "="*50)
            self.train_one_epoch()
            self.validate()  # perform validation or testing
            self.current_epoch += 1



    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        train_elements = 0

        for batch_idx, data in enumerate(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(q_list, a_list, l_list, d_list)  # predicted student performance

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])  # real student performance

            output = torch.masked_select(output, target_masks_list[:, 2:]) # masked attempt of non-assessed activities
            loss = self.criterion(output.float(), label.float())

            self.train_loss += loss.item()
            train_elements += target_masks_list[:, 2:].int().sum()
            loss.backward()  # compute the gradient

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm) # clip gradient to
            # avoid gradient vanishing or exploding
            self.optimizer.step()  # update the weight
            # self.scheduler.step()  # for CycleLR Scheduler or MultiStepLR
            self.current_iteration += 1

        self.train_loss = self.train_loss / train_elements
        self.scheduler.step(self.train_loss)
        self.train_loss_list.append(self.train_loss)
        self.logger.info("Train Loss: {:.6f}".format(self.train_loss))
        print("Train Loss: {:.6f}".format(self.train_loss))



    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
            # print("Validation Result at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
            # print("Test Result at Epoch: {}".format(self.current_epoch))
        test_loss = 0
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_list_l = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)

                output = self.model(q_list, a_list, l_list, d_list)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])

                test_loss += self.criterion(output.float(), label.float()).item()
                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())
                # print(list(zip(true_labels, pred_labels)))
                test_elements = target_masks_list[:, 2:].int().sum()
                test_loss = test_loss/ test_elements
                print("Test Loss: {:.6f}".format(test_loss))
                self.test_loss_list.append(test_loss)
        self.track_best(true_labels, pred_labels)  # calculate the test evaluation metric and check whether the current
        # test metric is the best, save learned model if it is the best




    def track_best(self, true_labels, pred_labels):
        self.pred_labels = np.array(pred_labels).squeeze()
        self.true_labels = np.array(true_labels).squeeze()
        self.logger.info(
            "pred size: {} true size {}".format(self.pred_labels.shape, self.true_labels.shape))
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
            perf_mae = metrics.mean_absolute_error(self.true_labels, self.pred_labels)
            self.logger.info('RMSE: {:.05}'.format(perf))
            print('RMSE: {:.05}'.format(perf))
            self.logger.info('MAE: {:.05}'.format(perf_mae))
            print('MAE: {:.05}'.format(perf_mae))
            if perf < self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
                torch.save(self.model.state_dict(),
                           'saved_model/{}/{}/sl_{}_eq_{}_ea_{}_el_{}_h_{}_fold_{}.pkl'.format(self.config.data_name,
                                                                                               self.config.model_name,
                                                                                               self.config.max_seq_len,
                                                                                               self.config.embedding_size_q,
                                                                                               self.config.embedding_size_a,
                                                                                               self.config.embedding_size_l,
                                                                                               self.config.hidden_size,
                                                                                               self.config.fold))

            self.test_rmse_list.append(perf)
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(self.true_labels, self.pred_labels)
            pr_auc = metrics.auc(rec, prec)
            self.logger.info('ROC-AUC: {:.05}'.format(perf))
            print('ROC-AUC: {:.05}'.format(perf))
            self.logger.info('PR-AUC: {:.05}'.format(pr_auc))
            print('PR-AUC: {:.05}'.format(pr_auc))
            if perf > self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
                torch.save(self.model.state_dict(),
                           'saved_model/{}/{}/sl_{}_eq_{}_ea_{}_el_{}_h_{}_fold_{}.pkl'.format(self.config.data_name,
                                                                                          self.config.model_name,
                                                                                          self.config.max_seq_len,
                                                                                          self.config.embedding_size_q,
                                                                                          self.config.embedding_size_a,
                                                                                          self.config.embedding_size_l,
                                                                                          self.config.hidden_size,
                                                                                          self.config.fold))

            self.test_roc_auc_list.append(perf)
            self.test_pr_auc_list.append(pr_auc)
        else:
            raise AttributeError


