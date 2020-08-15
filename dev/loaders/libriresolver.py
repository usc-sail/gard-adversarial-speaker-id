from pathlib import Path

class LibriSpeechSpeakers():
    """ implemented for our purposes using index 0..251 """

    def __init__(self, root, subset):
        root = Path(root)
        with open(root / "LibriSpeech" / "SPEAKERS.TXT") as fp:
            lines = [line for line in fp.readlines() if line[0] != ";"]
            lines = [line.split("|") for line in lines]
            self.gender = {int(x[0]): x[1].strip() for x in lines}
            self.speakers = [
                int(x[0]) for x in lines if x[2].strip() == subset]

    def get_gender(self, i):
        # assert
        return self.gender[self.speakers[i]]

    def get_num_speakers(self):
        return len(self.speakers)
