import argparse
from argparse import ArgumentParser
import logging
import time

import torch
import torch.nn as nn

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.models import RawAudioCNN

from hparams import hp
import pdb, sys, os
from art.classifiers import PyTorchClassifier
from dev.factories import AttackerFactory
from dev.defences import EnsembleAdversarialTrainer 
from art.attacks import FastGradientMethod

def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")

def resolve_attacker_args(args, eps, eps_step, max_iter=None):
    targeted = False
    if args.attack == "NoiseAttack":
        kwargs = {"eps": eps}
    elif args.attack == "FastGradientMethod":
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted}
    elif args.attack == "ProjectedGradientDescent":
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted, "max_iter":max_iter}
    else:
        raise NotImplementedError
    return kwargs


def main(args):

    n_epochs = args.num_epochs

    # Step 0: parse args and init logger
    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    generator_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True
    }

    # Step 1: load data set
    data_resolver = LibriSpeechSpeakers(hp.data_root, hp.data_subset)
    logging.info("Number of speakers (classes) ={}".format(data_resolver.get_num_speakers()))

    train_data = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        subset="train",
        project_fs=hp.sr,
        wav_length=args.wav_length,
    )
    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    if args.model_type=='cnn':
        model = RawAudioCNN(num_class=data_resolver.get_num_speakers())
    else:
        logging.error('Please provide a valid model architecture type!')
        sys.exit(-1)
        
    print(model)
    
   #multi-gpu code, if needed in future
   #if torch.cuda.device_count() > 1:
   #    print("Let's use", torch.cuda.device_count(), "GPUs!")
   #    model = nn.DataParallel(model)
 
    if _is_cuda_available():
        model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
    elif args.optimizer=='adam':
        print()
        print('Using Adam optimizer\n')
        optimizer = torch.optim.Adam(model.parameters())
       
    #wrap model with ART classifier class
    model.train()
    classifier_art = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=[1, 5 * hp.sr],  # FIXME
        nb_classes=data_resolver.get_num_speakers(),
    )
    
    eps = args.epsilon
    kwargs = resolve_attacker_args(args, eps, eps_step=eps / 5, max_iter=args.attack_max_iter) 
    attacker = AttackerFactory()(args.attack)(classifier_art, **kwargs)

    #adversarial training
    adv_trainer = EnsembleAdversarialTrainer(classifier=classifier_art, attack_methods=[attacker], ratio=args.ratio, augment=args.augment)
    adv_trainer.fit_generator(train_generator, args.num_epochs)
    logging.info("Finished Training")

    # Step 4: save model
    if args.model_ckpt is None:
        ckpt = f"model/libri_model_raw_audio_{time.strftime('%Y%m%d%H%M')}.pt"
    else:
        ckpt = args.model_ckpt

    ckpt_optim = os.path.join(os.path.dirname(ckpt), os.path.basename(ckpt)[:-3]+'_optimizer.pt')

    #save model
    torch.save(adv_trainer.get_backend_model(), ckpt)
    torch.save(adv_trainer.get_backend_optimizer(), ckpt_optim)
   #multi-gpu code, if needed in future
   #torch.save(adv_trainer.get_backend_model().module, ckpt)


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset", \
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", "--model_ckpt", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "-mt", "--model_type", type=str, default='cnn', help="Model type: cnn")
    parser.add_argument(
        "-e", "--epsilon", type=float, default=0.002,
        help="Perturbation scale")
    parser.add_argument(
        "-mi", "--attack_max_iter", type=int, default=None,
        help="Maximum number of iterations in attack algorithm")
    parser.add_argument(
        "-l", "--wav_length", type=int, default=80_000,
        help="Max length of waveform in a batch")
    parser.add_argument(
        "-epochs", "--num_epochs", type=int, default=30,
        help="Number of epochs")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=128,
        help="batch size")
    parser.add_argument(
        "-opt", "--optimizer", type=str, default='adam',
        help="optimizer: sgd, adam.")
    parser.add_argument(
        "-aug", "--augment", type=int, default=0,
        help="Gaussian noise augmentation to normal samples")
    parser.add_argument(
        "-r", "--ratio", type=float, default=0.5, help="Proportion of adversarial samples, 1=train only on adversarial samples")
    parser.add_argument(
        "-g", "--log", type=str, default=None, help="log file", required=True)
    parser.add_argument("-a", "--attack", default="ProjectedGradientDescent", \
            help="ProjectedGradientDescent, FastGradientMethod") 

    args = parser.parse_args()

    with open(args.log, 'w') as f:
        f.write("")

    return args


if __name__ == "__main__":
    main(parse_args())
