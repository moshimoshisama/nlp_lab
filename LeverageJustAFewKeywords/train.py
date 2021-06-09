import json
import logging
from os import name
import time
import torch
import os

from torch import nn, optim
from torch.utils import data
from torch.utils.data import sampler
from tqdm import tqdm

from dataset import TestDataset, Dataset
from model import Student, Teacher

from sklearn.metrics import f1_score


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
torch.set_printoptions(profile="full")
logging.getLogger('transformers').setLevel(logging.WARNING)

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p, q):
        b = q * torch.log(p)
        b = -1. * b.sum()

        return b


class Trainer:
    def __init__(self, hparams, device) -> None:
        self.hparams = hparams
        self.device = device
        self.asp_cnt = self.hparams['student']['num_aspect']
        self.start = time.time()
        logging.info('loading dataset...')
        self.ds = Dataset(hparams['aspect_init_file'], hparams['train_file'],
                          hparams['student']['pretrained'], hparams['maxlen'])
        test_ds = TestDataset(
            hparams['aspect_init_file'], hparams['test_file'])
        self.test_loader = data.DataLoader(test_ds, batch_size=500, num_workers=2)  # colab warning for 2 workers

        logging.info(f'dataset_size: {len(self.ds)}')

        logging.info('loading model...')
        self.idx2asp = torch.tensor(self.ds.get_idx2asp()).to(self.device)
        self.teacher = Teacher(self.idx2asp, self.asp_cnt,
                               self.hparams['general_asp'], self.device).to(self.device)
        self.student = Student(hparams['student']).to(self.device)
        self.student_opt = optim.Adam(self.student.parameters(), hparams['lr'])
        self.criterion = EntropyLoss().to(self.device)
        # self.criterion = nn.BCELoss(reduction='sum').to(self.device)

        # self.z = self.reset_z()
        self.reset_z()
        logging.debug(f'__init__: {time.time() - self.start}')

    def save_model(self, path, model_name):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.hparams, f)
        torch.save({'teacher_z': self.z, 'student': self.student.state_dict()}, os.path.join(
            path, f'epoch_{model_name}_student.pt'))

    def train_loader(self, ds):
        # sampler = data.RandomSampler(ds, replacement=True, num_samples=10000)
        return data.DataLoader(ds, batch_size=self.hparams['batch_size'], num_workers=2)    # colab warning for 2 workers, shuffle=False

    def train(self, epochs, inner_loop=3):
        prev_best = torch.tensor([-1])
        loader = self.train_loader(self.ds)
        loss_list = []
        score_list = []
        agr_ratio_list = []
        self.reset_z()
        for epoch in range(epochs):
            logging.info(f'Epoch: {epoch}')
            # loss = self.train_per_epoch(loader)
            loss, agr_ratio = self.train_per_epoch_ISWD(loader, inner_loop)
            score = self.test()
            loss_list.append(loss)
            score_list.append(score)
            agr_ratio_list.append(agr_ratio)
            logging.info(f'epoch: {epoch}, f1_mid: {score:.3f}, prev_best: {prev_best.item():.3f}')
            if prev_best < score:
                self.save_model(self.hparams['save_dir'], f'{epoch}')
                prev_best = score
        for i in range(len(loss_list)):     # for easy result-checking at the very end of training
            logging.info("epoch {}:\tloss: {:.3f}\tscore: {:.3f}\tagree_ratio: {:.3f}".format(
                            i, loss_list[i], score_list[i], agr_ratio_list[i]))

    def train_per_epoch_ISWD(self, loader, inner_loop=3):
        # pbar = tqdm(total=len(loader))
        loss_out_loop = []
        for i in range(inner_loop):
            loss_ep = []
            # train student
            train_bar = tqdm(total=len(loader))
            for x_bow, x_id in loader:
                x_bow, x_id = x_bow.to(self.device), x_id.to(self.device)
                self.student_opt.zero_grad()
                t_logits = self.teacher(x_bow, self.z)
                s_logits = self.student(x_id)
                loss = self.criterion(s_logits, t_logits)
                loss_ep.append(loss)
                loss.backward()
                self.student_opt.step()
                train_bar.update(1)
                train_bar.set_description(f'loss: {loss:.3f}')
            train_bar.close()
            aver_loss = sum(loss_ep) / len(loss_ep)
            loss_out_loop.append(aver_loss)
            # student converge
            if i > 0 and abs(loss_out_loop[i-1] - aver_loss) / aver_loss < 0.005:
                break
        # inference with student
        s_logits_ep = []
        x_bow_ep = []
        for x_bow, x_id in tqdm(loader):
            x_bow, x_id = x_bow.to(self.device), x_id.to(self.device)
            with torch.no_grad():
                s_logits = self.student(x_id)
            s_logits_ep.append(s_logits)
            x_bow_ep.append(x_bow)
        # calculate z
        s_logits_ep = torch.cat(s_logits_ep, dim=0)
        x_bow_ep = torch.cat(x_bow_ep, dim=0)
        self.z = self.calc_z(s_logits_ep, x_bow_ep)
        # update teacher
        t_logits_ep = self.teacher(x_bow_ep, self.z)
        # TODO use old or new teacher to compare teacher and student?
        agreement_ratio = (s_logits_ep.max(-1)[1] == t_logits_ep.max(-1)[1]).sum() / len(t_logits_ep)
        print(f"{agreement_ratio*100:.2f}% of samples have same result from student and teacher.")
        return aver_loss, agreement_ratio
    
    def train_per_epoch(self, loader):
        losses = []
        pbar = tqdm(total=len(loader))
        self.reset_z()

        for x_bow, x_id in loader:
            x_bow, x_id = x_bow.to(self.device), x_id.to(self.device)
            loss = self.train_step(x_bow, x_id)
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_description(f'loss:{loss.item():.3f}')
        pbar.close()
        losses = sum(losses) / len(losses)
        logging.info(f'train_loss: {losses}')
        return losses

    def reset_z(self):
        self.z = torch.ones(
            (self.asp_cnt, len(self.ds.asp2id))).to(self.device)    # [asp_cnt, bow_size]
        self.z_sum = torch.ones(len(self.ds.asp2id)).to(self.device)
        # return torch.softmax(z, dim=-1)
        # return z
    
    def train_step(self, x_bow, x_id):
        # TODO: apply the Iterative Seed Word Distillation to each batch???
        n_step = 3
        # apply teacher
        # self.z = self.reset_z()     # TODO: when to reset z
        t_logits = self.teacher(x_bow, self.z)  # [B, asp_cnt]
        loss = 0.
        prev = -1
        print()
        weights = self.build_weights(n_step)
        for i in range(n_step):
            # train student Eq. 2
            self.student_opt.zero_grad()
            s_logits = self.student(x_id)   # [B, asp_cnt]
            loss = self.criterion(s_logits, t_logits)
            # print(f'bow: {x_bow}')
            # print(f'z: {self.z}')
            # print(f'teacher:{t_logits.max(-1)[1]}')
            # print(f'x_id{x_id}')
            # print(f'student:{s_logits.max(-1)[1]}')
            loss.backward()
            self.student_opt.step()
            tmp = (t_logits.max(-1)[1] == s_logits.max(-1)[1]).sum()    # number of coincide
            if tmp == prev or tmp.item() == t_logits.shape[0]:
                break
            prev = tmp
            # update teacher Eq.4
            # self.z = self.calc_z(s_logits, x_bow)
            self.z, self.z_sum = self.calc_z_accumulate(s_logits, x_bow, weights[i])

            # apply teacher Eq. 3
            t_logits = self.teacher(x_bow, self.z)
        return loss

    def test(self):
        score = []
        result = []
        ground = []
        self.student.eval()
        for batch in self.test_loader:
            idx, bow, labels = batch
            idx = idx.to(self.device)
            bow = bow.to(self.device)
            with torch.no_grad():
                logits = self.student(idx)
            result.append(logits.max(-1)[1].detach().cpu())
            ground.append(labels.max(-1)[1])

        result = torch.cat(result, dim=0)
        ground = torch.cat(ground, dim=0)
        score = f1_score(ground.numpy(), result.numpy(), average='micro')
        self.student.train()
        return score

    def calc_z(self, logits, bow):
        """z
        Args:
            logits: B, asp_cnt
            bow: B, bow_size
        Returns:
            z: asp_cnt, bow_size
        """
        val, idx = logits.max(1)    # [B]
        num_asp = logits.shape[1]
        r = torch.stack([(bow[torch.where(idx == k)] > 0).float().sum(0)
                         for k in range(num_asp)])  # [asp_cnt, bow_size], default dim=0
        # change the way of summation, unofficial repo wrong, because after changing performance gets better
        # bsum = r.sum(-1).view(-1, 1)    # [asp_cnt, 1]
        bsum = r.sum(0).view(1, -1)     # [1, bow_size]
        bsum = bsum.masked_fill(bsum == 0., 1e-10)
        z = r / bsum
        # z = torch.softmax(r, -1)
        # print(f'z: {z}')
        return z

    def calc_z_accumulate(self, logits, bow, weight):
        val, idx = logits.max(1)
        num_asp = logits.shape[1]
        r = torch.stack([(bow[torch.where(idx == k)] > 0).float().sum(0) for k in range(num_asp)])
        accu_r = self.z * self.z_sum + r * weight
        accu_r_sum = accu_r.sum(0).view(1, -1)
        accu_r_sum = accu_r_sum.masked_fill(accu_r_sum == 0., 1e-10)
        z = accu_r / accu_r_sum
        return z, accu_r_sum

    @staticmethod
    def build_weights(n_step):
        '''
        geometric sequence, ratio=2
        '''
        weights = torch.arange(n_step)
        weights = torch.pow(2, weights) / (2 ** len(weights) - 1)
        return weights
