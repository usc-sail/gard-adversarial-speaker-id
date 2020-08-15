""" 
0: wav (in [-1, 1])
1: fs
2: text (str)
3: speaker id
4: chapter id
5: utt id


Let's reserve 10% speakers for testing (?) for the remaining, reserve 10% utt of each speaker.
"""

import torch
from torchaudio.datasets import LIBRISPEECH
from warnings import warn


URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"


class LibriSpeech4SpeakerRecognition(LIBRISPEECH):
    def __init__(
        self, root, project_fs, subset, wav_length=None, url=URL, 
        folder_in_archive=FOLDER_IN_ARCHIVE, download=False,
        train_speaker_ratio=1, train_utterance_ratio=0.9,
        return_file_name=False
    ):
        """
        subset: "train", "test", "outside"
        """
        super().__init__(root, url=url, folder_in_archive=folder_in_archive, download=download)
        self._split(subset, train_speaker_ratio, train_utterance_ratio)
        self.project_fs = project_fs
        self.wav_length = wav_length
        self.return_file_name = return_file_name

    def _split(self, subset, train_speaker_ratio, train_utterance_ratio):
        """ Splits into training and testing sets """
        def _parse(name_string):
            speaker_id, chapter_id, utterance_id = name_string.split("-")
            return speaker_id, chapter_id, utterance_id

        n_total_utterance = len(self._walker)

        self._walker.sort()

        utt_per_speaker = {}
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            if utt_per_speaker.get(speaker_id, None) is None:
                utt_per_speaker[speaker_id] = [utterance_id]
            else:
                utt_per_speaker[speaker_id].append(utterance_id)
        
        speakers = list(utt_per_speaker.keys())
        speakers.sort()
        num_train_speaker = int(len(speakers) * train_speaker_ratio)
        speakers = {
            "train": speakers[:num_train_speaker],
            "test": speakers[num_train_speaker:]
        }
        self.speakers = {
            "train": [int(s) for s in speakers["train"]],
            "test": [int(s) for s in speakers["test"]],
        }
        
        for spk in speakers["train"]:
            utt_per_speaker[spk].sort()
            num_train_utterance = int(len(utt_per_speaker[spk]) * train_utterance_ratio)
            utt_per_speaker[spk] = {
                "train": utt_per_speaker[spk][:num_train_utterance],
                "test": utt_per_speaker[spk][num_train_utterance:],
            }
        
        trn_walker = []
        test_walker = []            
        outsiders = []
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            if speaker_id in speakers["train"]:
                if utterance_id in utt_per_speaker[speaker_id]["train"]:
                    trn_walker.append(filename)
                else:
                    test_walker.append(filename)
            else:
                outsiders.append(outsiders)

        if subset == "train":
            self._walker = trn_walker
        elif subset == "test":
            self._walker = test_walker
        else:
            self._walker = outsiders

        order = "first" if subset == "training" else "last"
        warn(
            f"Deterministic split: {len(self._walker)} out of the {order}"
            f" {n_total_utterance} utterances are taken as {subset} set.",
            UserWarning
        )

    def __getitem__(self, n):
        if not self.return_file_name:
            waveform, sample_rate, _, speaker_id, _, _ = super().__getitem__(n)
        else:
            waveform, sample_rate, _, speaker_id, chapter_id, utt_id = super().__getitem__(n)

        n_channel, duration = waveform.shape
        if self.wav_length is None:
            pass
        elif duration > self.wav_length:
            i = torch.randint(0, duration - self.wav_length, []).long()
            waveform = waveform[:, i: i + self.wav_length]
        else:
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(n_channel, self.wav_length - duration)
                ],
                1
            )

        # waveform = librosa.core.resample(waveform, sample_rate, self.project_fs)
        if not self.return_file_name:
            return waveform, self.speakers["train"].index(speaker_id)
        else:
            return waveform, self.speakers["train"].index(speaker_id), str(speaker_id)+"-"+str(chapter_id)+"-"+str(utt_id).zfill(4)

        # fileid = self._walker[n]
        # return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)
