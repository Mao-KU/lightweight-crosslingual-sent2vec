#coding:utf8
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from shutil import copyfile
import random
from io import open
import os
from torch.autograd import Variable
import numpy as np
import argparse
import torch
from torch import nn,optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import detect_anomaly
from tqdm import tqdm, trange
from modeling import BilingualModel
from preprocessing import DataProvider
from modeling.optimization import BertAdam, warmup_linear
from torch.utils.data import Dataset
import random
from utils import *
from config import config 
import logging
import time
from datetime import datetime
from torch.cuda import amp
import pytorch_warmup as warmup
from thop import profile


logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

        
def evaluate(val_loader, model, epoch, device):
    val_progressor = ProgressBar(mode="Vali",\
                                 epoch=epoch,\
                                 total_epoch=config.num_train_epochs,\
                                 model_name=config.model_name,\
                                 total=len(val_loader))
    model.to(device)
    losses = AverageMeter()
    avg_recall = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_progressor.current = i 
            batch = tuple(t.to(device) for t in batch)
            sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la = batch
            loss, recall = model(sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la)
            losses.update(torch.mean(loss).item(), sent1_ids.size(0))
            avg_recall.update(torch.mean(recall).item(), sent1_ids.size(0))
            val_progressor.current_loss = losses.val
            val_progressor.avg_loss = losses.avg
            val_progressor.cur_time = str(datetime.now().strftime('%d %H:%M:%S'))
            if i%100==0:
                val_progressor()
        val_progressor.done()
    return losses.avg


def main():
    if not config.resume:
        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)
        if os.path.exists(os.path.join(config.output_dir, config.model_name)):
            if os.path.exists(os.path.join(config.output_dir, config.model_name, config.best_models)) and os.listdir(os.path.join(config.output_dir, config.model_name, config.best_models)):
                logger.error('Exists best checkpoints in {}. Please Check.'.format(os.path.join(config.output_dir, config.model_name,  config.best_models)))
                exit(0)
        if os.path.exists(os.path.join(config.output_dir, config.model_name)):
            shutil.rmtree(os.path.join(config.output_dir , config.model_name))
        os.mkdir(os.path.join(config.output_dir , config.model_name))
        os.mkdir(os.path.join(config.output_dir , config.model_name, config.best_models))
        shutil.copy(config.config_path, os.path.join(config.output_dir, config.model_name, 'config_'+config.model_name +'.py'))
        logger.info('Copying config.py to {}'.format(os.path.join(config.output_dir,config.model_name, 'config_'+config.model_name)))

    device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    
    model = BilingualModel(config.vocab_size, config)
    if torch.cuda.device_count()>1:
        model = model.cuda()
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, amsgrad=True)

    start_epoch = 0
    best_loss = np.inf
    best_loss_save = np.inf
    resume = config.resume
    
    if resume:
        if not os.path.exists(os.path.join(config.output_dir, config.model_name)):
            logger.error('No model found in path: {}'.format(os.path.join(config.output_dir, config.model_name)))
        checkpoint = torch.load(os.path.join(config.output_dir, config.model_name, config.best_models, 'model_best.pth.tar'))
        old_state = checkpoint['state_dict']
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(old_state)
        optimizer.load_state_dict(checkpoint["optimizer"])

    dataloader = DataProvider(config, True, 'train')
    train_data, validation_data = dataloader.data_loader, dataloader.vali_loader
    num_train_optimization_steps = int(dataloader.dataset.num_samples / config.train_batch_size) * config.num_train_epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.lr_decay_from], gamma=0.1)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    warmup_scheduler.last_step = -1     
    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Model Name = %s", config.model_name)
    logger.info("  bpe_model_path = %s", config.bpe_path)
    logger.info("  Num examples = %d", train_data.dataset.num_samples)
    logger.info("  Batch size = %d", config.train_batch_size)
    logger.info("  Num steps per epoch = %d", num_train_optimization_steps/config.num_train_epochs + 1)
    logger.info("  Has FC: %s", str(config.has_FC))
    logger.info("  Has sentence-alignment loss: %s", str(config.has_sentence_loss))
    logger.info("  Has sentence-similarity loss: %s", str(config.has_sentence_similarity_loss))
    model.train()
    scaler = amp.GradScaler()
    
    for epoch in range(start_epoch, int(config.num_train_epochs)):
        logger.info('Start new epoch {}'.format(str(epoch)))
        tr_loss = AverageMeter()
        train_progressor = ProgressBar(mode="Train",\
                                       epoch=epoch,\
                                       total_epoch=int(config.num_train_epochs),\
                                       model_name=config.model_name,\
                                       total=len(train_data))
        for step, batch in enumerate(train_data):
            lr_scheduler.step(epoch)
            warmup_scheduler.dampen()
            batch = tuple(t.to(device) for t in batch)
            sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la = batch
            with amp.autocast():
                loss, recall = model(sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la)
            tr_loss.update(torch.mean(loss).item(), batch[0].size(0))
            train_progressor.current =  step + 1
            train_progressor.current_loss = tr_loss.val
            train_progressor.avg_loss = tr_loss.avg
            train_progressor.cur_time = str(datetime.now().strftime('%d %H:%M:%S'))
            if step % 10000 == 0:
                train_progressor()
                logger.info("Token Classification Acc: Not Defined")
            optimizer.zero_grad()
            scaler.scale(torch.mean(loss)).backward()
            scaler.step(optimizer)
            global_step += 1
            scaler.update()
            
        train_progressor.done()
        eval_loss = loss
        if config.has_validation:
            valid_loss = evaluate(validation_data, model, epoch, device)
            eval_loss =  valid_loss
        is_best = eval_loss <= best_loss
        best_loss = min(eval_loss, best_loss)
        try:
            best_loss_save = best_loss.cpu().data.numpy()
        except:
            pass
        logger.info('Trying to save model.')
        save_checkpoint({
                    "epoch": epoch + 1,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "valid_loss": eval_loss,
        },is_best,epoch)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='modelName', default='', help='modelName')
    parser.add_argument('-r', dest='resume', default=False, help='resume training')
    parser.add_argument('-t', dest='is_train', default=True, help='is train?')
    parser.add_argument('-la', dest='target_la', default='fr', help='target language')
    args = parser.parse_args()    
    config.set_config(args.modelName, args.resume, args.is_train)
    main()

