from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass
class RawDataConfig:
    sampling_rate: int = 44100

    train = {'csv': './data/train.csv',
             'wav': './data/audio_train'}

    test = {'csv': './data/sample_submission.csv',
            'wav': './data/audio_test'}

    labels: list = field(init=False)

    def __post_init__(self):
        self.labels = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock', 'Gunshot_or_gunfire',
                       'Clarinet', 'Computer_keyboard', 'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
                       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak',
                       'Scissors', 'Harmonica', 'Gong', 'Microwave_oven', 'Burping_or_eructation', 'Double_bass',
                       'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano', 'Meow',
                       'Drawer_open_or_close', 'Applause', 'Acoustic_guitar', 'Violin_or_fiddle', 'Finger_snapping']


@dataclass
class LogMelData:
    sampling_rate: int = 22050
    n_mels: int = 64
    frame_weight: int = 80
    frame_shift: int = 10
    n_fft: int = field(init=False)
    hop_length: int = field(init=False)

    def __post_init__(self):
        self.n_fft = int(self.frame_weight / 1000 * self.sampling_rate)
        self.hop_length = int(self.frame_shift / 1000 * self.sampling_rate)


@dataclass
class DataLoader:
    sampling_rate: int = 22050
    audio_duration: float = 1.5
    frame_shift: int = 10
    input_length: int = field(init=False)

    batch_size = 128
    seed = 3
    n_folds = 5

    method: str = 'logmel'

    train_transform: str = 'base'
    val_transform: str = 'base'
    test_transform: str = 'base'

    train_aug: str = 'logmel'
    val_aug: str = 'logmel'
    test_aug: str = 'logmel'

    def __post_init__(self):
        self.input_length = int(self.audio_duration * 1000 / self.frame_shift)


@dataclass
class Model:
    arch: str = 'resnet50'
    num_classes: int = 41
    pretrained: bool = True


@dataclass
class RetrainOnVal:
    flag: bool = True

    min_epochs: int = 2
    max_epochs: int = 10
    n_epochs: int = 6

    mixup: bool = True
    mixup_p: float = 0.7

    device: str = 'cuda'

    grad_accum_batch = {3: 2}


@dataclass
class TrainModel:
    seed: int = 3

    min_epochs: int = 5
    max_epochs: int = 60
    n_epochs: int = 50

    mixup: bool = True
    mixup_p: float = 0.7

    optim: dict = field(init=False)

    device: str = 'cuda'

    grad_accum_batch = {5: 2, 25: 3}

    on_folds: list = field(init=False)
    retrain_val: RetrainOnVal = RetrainOnVal()

    def __post_init__(self):
        self.optim = {
            'optimizer': {
                'sgd': {'lr': 0.009, 'momentum': 0.9, 'weight_decay': 0.0005},
                'adam': {'lr': 0.008, 'weight_decay': 0}
            },
            'scheduler': {
                'cosine_annealing': {'T_max': self.n_epochs},
                'multi_step_lr': {'milestones': [20, 30, 40], 'gamma': 0.7}
            }
        }
        self.on_folds = [1, ]


@dataclass_json
@dataclass
class DefaultConfig:
    rawdata: RawDataConfig = RawDataConfig()

    logmel: LogMelData = LogMelData()
    method: str = 'logmel'

    dataloader: DataLoader = DataLoader()

    model: Model = Model()

    train: TrainModel = TrainModel()

