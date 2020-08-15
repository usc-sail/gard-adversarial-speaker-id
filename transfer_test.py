from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pathlib import Path

from art.classifiers import PyTorchClassifier

from dev.factories import AttackerFactory
from dev import attacks as attack_dev
from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.transforms import Preprocessor

from hparams import hp
import os, pdb


# set seed
np.random.seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_attacker_args(args, eps, eps_step, norm=np.inf, max_iter=None):
    targeted = False if args.target is None else True
    if args.attack == "DeepFool":
        kwargs = {"epsilon": eps}
    elif args.attack == "NoiseAttack":
        kwargs = {"eps": eps}
    elif args.attack == "FastGradientMethod":
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted, "norm":norm}
    elif args.attack == "ProjectedGradientDescent":
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted, "norm":norm, "max_iter":max_iter}
    elif args.attack  in ["CarliniLInfMethod"]:
        kwargs = {"eps": eps, "targeted": targeted}
    elif args.attack  in ["CarliniL2Method"]:
        kwargs = {"confidence": eps, "targeted": targeted}
    else:
        raise NotImplementedError
    return kwargs


class AttackResultCounter():
    """
    Compute targeted and non-targeted attack success rate (ASR)
    Definition: 
        [TASR](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack/overview/evaluation)
        [NASR](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/overview/evaluation)
    
    Note that there is a caveat in this def of NASR: 
        if your model always predict the wrong label,
        without attacks, NASR is already 100%.
    """
    def __init__(self):
        self.n_instance = 0
        self.n_nontarget_instance = 0
        self.n_correct_prediction = 0
        self.n_successful_untargeted = 0
        self.n_successful_targeted = 0
        self.n_correct_prediction_adversarial = 0
    
    def update(self, label, prediction, adversarial_prediction, target=None):
        self.n_instance += 1

        if prediction == label:
            self.n_correct_prediction += 1

        if adversarial_prediction == label:
            self.n_correct_prediction_adversarial += 1
            #this is simply accuracy on adversarial samples
            #much simpler to use

        if prediction != adversarial_prediction:
            self.n_successful_untargeted += 1
            # TODO: this is the def NIPS'17 used
            ##  This is somewhat misleading, it can be high if 
            #   the model predicts same WRONG label for normal
            #   sample and its adversarial counterpart

        if target is not None:
            if label != target:
                self.n_nontarget_instance += 1
                if adversarial_prediction == target:
                    self.n_successful_targeted += 1

    def asr(self):
        return self.n_successful_untargeted / self.n_instance

    def tasr(self):
        if self.n_nontarget_instance == 0:
            return np.nan
        else:
            return self.n_successful_targeted / self.n_nontarget_instance

    def accuracy(self):
        return self.n_correct_prediction / self.n_instance

    def accuracy_adversarial(self):
        return self.n_correct_prediction_adversarial / self.n_instance


class ProjectorTSVWriter():
    def __init__(self, output_dir, resolver):
        self.meta = open(output_dir / "meta.tsv", "w")
        self.meta.write(
            "\t".join(["Index", "TrueGender", "Prediction", "PredictedGender", "Source"]) + "\n"
        )

        self.vec = open(output_dir / "vec.tsv", "w")
        self.resolver = resolver

    def update(self, embedding, adv_embedding, label, pred, pred_adv, i):
        stem = f"{i:04d}-{label:03d}\t{self.resolver.get_gender(label)}"

        self.vec.write("\t".join([f"{x.item():.8f}" for x in embedding]) + "\n")
        self.meta.write(
            f"{stem}\t{pred:03d}\t{self.resolver.get_gender(pred)}\toriginal\n")

        self.vec.write("\t".join([f"{x.item()}" for x in adv_embedding]) + "\n")
        self.meta.write(
            f"{stem}\t{pred_adv:03d}\t{self.resolver.get_gender(pred_adv)}\tadversarial\n")

    def close(self):
        self.meta.close()
        self.vec.close()


class AsyncReporter():
    def __init__(self, report_file, args, counter, num_params):
        if not Path(report_file).exists():
            header = (
                f"Attacker={args.attack}\n"
                f"Accuracy={counter.accuracy(): .4f}\n"
                f"Model={args.model_ckpt} (#parameters={num_params / 1e6:.3f} million)\n"
                "| avg eps    | avg SNR   |  ASR    |\n"
                "| ------     |   ---     |  ---    |\n"
            )
            with open(args.report, "a") as fp:
                fp.write(header)

        self.report_file = report_file
    
    def update(self, average_epsilon, average_snr, rate):
        with open(self.report_file, "a") as fp:
            row = (
                f"| {average_epsilon:.4e} |   "
                f"{average_snr:5.2f}   | "
                f"{rate: .4f} |\n"
            )
            fp.write(row)


def main(args):
    # load Libri test set
    resolver = LibriSpeechSpeakers(hp.data_root, hp.data_subset)
    return_file_name=False
    if args.save_wav:
        return_file_name = True
    dataset = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        subset="test",
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        project_fs=hp.sr,  # FIXME: unused
        wav_length=None,
        return_file_name=return_file_name
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)


    # load pretrained model
    model = (
        torch.load(args.model_ckpt)
        .eval()
        .to(device)
    )
    print(model)

    target_model = (
        torch.load(args.target_model_ckpt)
        .eval()
        .to(device)
    )
    print(target_model)

    # wrap model in a ART classifier
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, 5 * hp.sr],  # FIXME
        nb_classes=resolver.get_num_speakers(),
    )

    counter = AttackResultCounter()
    counter_target = AttackResultCounter()
    tsv_writer = ProjectorTSVWriter(args.output_dir, resolver=resolver)
    tsv_writer_target = ProjectorTSVWriter(args.output_dir, resolver=resolver)


    snrs = []
    epss = []
    for i, data in enumerate(loader, 1):

        if return_file_name:
            waveform, label, filename = data
            filename = filename[0]
        else:
            waveform, label = data
            

        if args.save_iter_id is not None and args.save_iter_id != i:
                print("Skipping iter ", i)
                continue

        #is this right? targeted attack is probably confused with attack with true labels/not
        #if args.target is None:
        #    y = None
        #else:
        #    y = 0 * label.numpy() + args.target

        #JATI
        if args.target is None:
            y = label #Original FGSM needs true labels to generate samples
                      #there is a label_leaking phenomenta too. See, the paper
                      #https://arxiv.org/pdf/1611.01236.pdf
                      #ADVERSARIAL MACHINE LEARNING AT SCALE, Alex Kurakin et. al.
                      #DON'T CONFUSE THIS WITH TARGETED ATTACK
        else:
            y = 0 * label.numpy() + args.target
        
        label = label.item()

        #this can be done outside for loop
        if args.epsilon is not None:
            eps = args.epsilon
        else:
            if args.norm == np.inf:
                eps = (waveform.pow(2).mean() / np.power(10, args.snr / 10)).sqrt().item()
            elif args.norm == 2:
                eps = (waveform.pow(2).sum() / np.power(10, args.snr / 10)).sqrt().item()
            elif args.norm == 1:
                eps = (waveform.abs().sum() / np.power(10, args.snr / 10 / 2)).item() / np.sqrt(np.log(10)) # resulting SNR is too small (10-20)  FIXME

        kwargs = resolve_attacker_args(args, eps, eps_step=eps / 5, norm=args.norm, max_iter=args.attack_max_iter)    # TODO ad-hoc
        
        # craft adversarial example with PGD        
        attacker = AttackerFactory()(args.attack)(classifier_art, **kwargs)
        adv_waveform = torch.from_numpy(
            attacker.generate(waveform, y=y)
        )

        noise = adv_waveform - waveform

        snr = 10 * (waveform.pow(2).mean() / noise.pow(2).mean()).log10()
        snrs.append(snr)
        epss.append(eps)

        # evaluate the classifier on the adversarial example
        with torch.no_grad():
            emb = model.encode(waveform.to(device))
            pred = (
                model.predict_from_embeddings(emb)
                .argmax(-1)
                .tolist()[0]
            )

            emb_target = target_model.encode(waveform.to(device))
            pred_target = (
                target_model.predict_from_embeddings(emb_target)
                .argmax(-1)
                .tolist()[0]
            )

            adv_emb = model.encode(adv_waveform.to(device))
            pred_adv = (
                model.predict_from_embeddings(adv_emb)
                .argmax(-1)
                .tolist()[0]
            )

            adv_emb_target = target_model.encode(adv_waveform.to(device))
            pred_adv_target = (
                target_model.predict_from_embeddings(adv_emb_target)
                .argmax(-1)
                .tolist()[0]
            )

        counter.update(label, pred, pred_adv, target=args.target)
        counter_target.update(label, pred_target, pred_adv_target, target=args.target)
        tsv_writer.update(emb[0], adv_emb[0], label, pred, pred_adv, i)
        tsv_writer_target.update(emb_target[0], adv_emb_target[0], label, pred_target, pred_adv_target, i)

        print(
            (
                f"Processing {i}/{len(loader)}: [L={label:3d}|P={pred:3d}|A={pred_adv:3d}], "
                f"SNR={snr:.2f} dB, "
                f"NASR={counter.asr():.4f}, "
                f"TASR={counter.tasr():.4f}, "
                f"accuracy={counter.accuracy():.4f}, "
                f"accuracy adv={counter.accuracy_adversarial():.4f}, "
                f"accuracy target={counter_target.accuracy():.4f}, "
                f"accuracy adv target={counter_target.accuracy_adversarial():.4f}"
            ),
            end="\r"
        )

        with open(args.log, 'a') as f:
            f.write("Processing "+str(i)+"/"+str(len(loader))+": [L="+str(label)+"|P="+str(pred)+"|A="+str(pred_adv)+"]\tSNR="+str(snr.item())+"\tNASR="+str(counter.asr())+"\tTASR="+str(counter.tasr())+"\taccuracy="+str(counter.accuracy())+"\taccuracy adversarial="+str(counter.accuracy_adversarial())+"\taccuracy target="+str(counter_target.accuracy())+"\taccuracy adversarial target="+str(counter_target.accuracy_adversarial())+"\n" )
        
        if args.save_wav:
            #print("Saving original and adversarial wavs for label =", label)
            print("Saving adversarial wav for label =", label)

            spk, chap, utt = filename.split("-")
            audio_save_dir = os.path.join(args.output_dir, "wavs", spk, chap)
            os.system("mkdir -p "+audio_save_dir)
            audio_save_file = os.path.join(audio_save_dir, filename+".wav")
            write(audio_save_file, hp.sr, 
                  adv_waveform[0, 0].detach().cpu().numpy())


    tsv_writer.close()
    print()


    # # =====================================================
    prep = Preprocessor()
    mel = prep(waveform[0])
    nel = prep(noise[0])
    ael = prep(adv_waveform[0])

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(
        f"Truth: {label}. Prediction: {pred}. Corrupted prediction: {pred_adv}\n"
        "Top: clean. Middle: noise. Bottom: noisy"
    )
    im = ax[0].imshow(mel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(nel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(ael[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[2])
    fig.savefig(args.output_dir / f"mel.png")

    fig, ax = plt.subplots(3, 1)
    im = ax[0].plot(waveform[0, 0].detach().cpu().numpy())
    im = ax[1].plot(noise[0, 0].detach().cpu().numpy())
    im = ax[2].plot(adv_waveform[0, 0].detach().cpu().numpy())
    fig.savefig(args.output_dir / f"wav.png")

    write(args.output_dir / f"noise.wav", hp.sr, noise[0, 0].detach().cpu().numpy())
    # write(args.output_dir / "ori.wav", hp.sr, waveform[0].numpy())
    # write(args.output_dir / "adv.wav", hp.sr, adv_waveform[0].numpy())

    if args.report is not None:
        reporter = AsyncReporter(
            args.report, args, counter, 
            num_params=count_parameters(model),
        )
        reporter.update(
            average_epsilon=np.mean(epss), 
            average_snr=np.mean([s for s in snrs if s < 100]),
            rate=counter.tasr() if args.target is not None else counter.asr(),
        )
    # # =====================================================


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset")
    parser.add_argument("-m", "--model_ckpt", required=True)
    parser.add_argument("-mtar", "--target_model_ckpt", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None, required=True)
    parser.add_argument("-a", "--attack", default="FastGradientMethod")
    parser.add_argument("-e", "--epsilon", type=float, default=None)
    parser.add_argument("-mi", "--attack_max_iter", type=int, default=None)
    parser.add_argument("-s", "--snr", type=float, default=None, help="signal-to-noise ratio (in decibel)")
    parser.add_argument(
        "-t", "--target", type=int, default=None,
        help="Attack target. Set it to `None` for untargeted attacks.")
    parser.add_argument(
        "-r", "--report", default=None,
        help="a text file for documenting the final results.")
    parser.add_argument(
        "-w", "--save_wav", type=int, default=0)
    parser.add_argument(
        "-wid", "--save_iter_id", type=int, help='only save at this iter id', default=None)
    parser.add_argument(
        "-g", "--log", type=str, default=None, help="log")
    parser.add_argument(
        "-n", "--norm", type=str, default="inf", help="inf, 1, 2")
    
    args = parser.parse_args()

    assert not (args.epsilon is None and args.snr is None), "Set either `epsilon` or `snr`"
    
    if args.norm == "inf":
        args.norm = np.inf
    else:
        args.norm = int(args.norm)

    args.save_wav = bool(args.save_wav)
    
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.log is None:
        args.log = os.path.join(args.output_dir, "log_"+time.strftime("%Y%m%d-%H%M%S")+".txt")
    with open(args.log, 'w') as f:
        f.write("")

    return args


if __name__ == "__main__":
    main(parse_args())
