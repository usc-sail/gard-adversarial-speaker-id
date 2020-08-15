import argparse
from argparse import ArgumentParser
import logging
import time

import torch
from torch.optim import SGD, Adam
import torch.nn.functional as F

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.models import RawAudioCNN, ALR, TDNN
from dev.utils import infinite_iter

from hparams import hp
import pdb, sys, os
import numpy as np


def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")


def noise_augmenter(inputs, labels, epsilon):
    """ Data augmentation with additive white uniform noise"""
    a = torch.rand([])
    noise = torch.rand_like(inputs)
    noise = noise.to(inputs.device)
    noise = 2 * a * epsilon * noise - a * epsilon
    noisy = inputs + noise
    inputs = torch.cat([inputs, noisy])
    labels = torch.cat([labels, labels])
    return inputs, labels


device = _get_device()


def main(args):  

    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    if args.model_ckpt is None:
        ckpt = f"model/libri_model_raw_audio_{time.strftime('%Y%m%d%H%M')}.pt"

    else:
        ckpt = args.model_ckpt

    generator_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers
    }

    # Step 1: load data set
    data_resolver = LibriSpeechSpeakers(hp.data_root, hp.data_subset)

    train_data = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        subset="train",
        project_fs=hp.sr,
        wav_length=args.wav_length,
    )
    train_generator = torch.utils.data.DataLoader(train_data, **generator_params)

    if args.model_type=='cnn':
        model = RawAudioCNN(num_class=data_resolver.get_num_speakers())
    elif args.model_type=='tdnn':
        model = TDNN(data_resolver.get_num_speakers())
    else:
        logging.error('Please provide a valid model architecture type!')
        sys.exit(-1)
        
    print(model)
        
    if _is_cuda_available():
        model.to(device)
        logging.info(device)


    alr = ALR()
    
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
    elif args.optimizer=='adam':
        print()
        print('Using Adam optimizer\n')
        optimizer = Adam(model.parameters(), lr=1e-3, betas=(.5, .999))


    # Step 3: train
    model.train()
    batch_idx = 0
    loss_epoch = []
    acc_epoch = []
    for batch_data in infinite_iter(train_generator):
        batch_idx += 1
        inputs, labels = (x.to(device) for x in batch_data)
        model.train()

        if args.epsilon > 0:
            inputs, labels = noise_augmenter(inputs, labels, args.epsilon)

        real_feature = model.encode(inputs)
        outputs = model.predict_from_embeddings(real_feature)
        class_loss = criterion(outputs, labels)

        if args.alr_weight > 0:
            input_adv = alr.get_adversarial_perturbations(model, inputs, labels)
            adv_feature = model.encode(input_adv)
            output_adv = model.predict_from_embeddings(adv_feature)
            alr_outputs = alr.get_alp(inputs, input_adv, outputs, output_adv, labels)
            alr_loss = alr_outputs[0].mean()
            loss = class_loss + args.alr_weight * alr_loss
            
            if args.cr_weight > 0:
                cr_loss = F.mse_loss(real_feature, adv_feature)
                loss += cr_loss
        else:
            loss = class_loss

        # Model computations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # training accuracy 
        acc_ = np.mean((torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy())
        loss_epoch.append(loss.item())
        acc_epoch.append(acc_)

        message = f"It [{batch_idx}] train-loss: {loss.item():.4f} \ttrain-acc (batch): {acc_:.4f}"
        
        if args.alr_weight > 0:
            message += f"\talr={alr_loss.item():.4e} "

        if args.cr_weight > 0:
            message += f"\tcr={cr_loss.item():.4e} "
        
        print(message, end="\r")
        logging.info(message)

        # Checkpointing
        if batch_idx % args.save_every == 0:
            torch.save(model, ckpt + ".tmp")
            torch.save(optimizer, ckpt + ".optimizer.tmp")
            print()

        # Termination
        if batch_idx > args.n_iters:
            break
        # Termination -- if n_epochs are provided
        if args.n_epochs is not None:
            done_ = batch_idx//len(train_generator)
            if batch_idx%len(train_generator)==0:
                msg_ = f"Epoch {done_}: loss = {np.mean(loss_epoch)} acc = {np.mean(acc_epoch)}"
                print(msg_)
                logging.info(msg_) 
                loss_epoch=[]
                acc_epoch=[]
            if done_ >= args.n_epochs:
                msg_ = "Finishing training based on n_epochs provided..."
                print(msg_)
                logging.info(msg_)
                break

    logging.info("Finished Training")
    torch.save(model, ckpt)


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset", \
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", "--model_ckpt", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "-g", "--log", type=str, default="train.log", help="Experiment log")
    parser.add_argument(
        "-mt", "--model_type", type=str, default='cnn', help="Model type: cnn or tdnn")
    parser.add_argument(
        "-l", "--wav_length", type=int, default=80000,
        help="Max length of waveform in a batch")
    parser.add_argument(
        "-n", "--n_iters", type=int, default=500000,
        help="Number of iterations for training"
    )
    parser.add_argument(
        "-ne", "--n_epochs", type=int, default=None,
        help="Number of epochs for training. Optional. Ignored if not provided."
    )
    parser.add_argument(
        "-s", "--save_every", type=int, default=10000, help="Save after this number of gradient updates"
    )
    parser.add_argument(
        "-e", "--epsilon", type=float, default=0,
        help="Noise magnitude in data augmentation; set it to 0 to disable augmentation")
    parser.add_argument(
        "-w", "--alr_weight", type=float, default=0,
        help="Weight of the adversarial Lipschitz regularizer"
    )
    parser.add_argument(
        "-c", "--cr_weight", type=float, default=0,
        help="Weight of consistency regularizer"
    )
    parser.add_argument(
        "-opt", "--optimizer", type=str, default='adam',
        help="Optimizer: sgd, adam")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=128,
        help="Batch size")
    parser.add_argument(
        "-nw", "--num_workers", type=int, default=8,
        help="Number of workers related to pytorch data loader")
    
    args = parser.parse_args()

    #clean log file
    with open(args.log, "w") as f:
        f.write("") 



    return args


if __name__ == "__main__":
    main(parse_args())
