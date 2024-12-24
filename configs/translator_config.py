class ConfigDataset():
    PATH_DATASET = "kde4"
    REVISION = None
    LANG1 = "en"
    LANG2 = "fr"


class ConfigModel():
    BATCH_SIZE = 8
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    MODEL_TOKENIZER = "Helsinki-NLP/opus-mt-en-fr"
    MODEL_NAME = "marian-finetuned-kde4-en-to-fr"
    TRAIN_SIZE = 0.9
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    METRICs = "sacrebleu"
    PATH_TENSORBOARD = "runs/data_run"
    PATH_SAVE = "translator"
    NUM_WARMUP_STEPS = 0

class ConfigHelper():
    TOKEN_HF = "xxx"
    AUTHOR = "Chessmen"