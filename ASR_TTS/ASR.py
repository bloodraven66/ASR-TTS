from speechbrain.pretrained import EncoderDecoderASR

class asr_model_info():
    def __init__(self):
        self.models = ['rnn', 'transformers', 'wav2vec']
        self.model_dependencies = {'rnn':['speechbrain/asr-crdnn-rnnlm-librispeech',
                                        'pretrained_models/asr-crdnn-rnnlm-librispeech'],
                                    'transformers':['speechbrain/asr-transformer-transformerlm-librispeech',
                                        'pretrained_models/asr-transformer-transformerlm-librispeech'],
                                    'wav2vec':['speechbrain/asr-wav2vec2-commonvoice-en',
                                        'pretrained_models/asr-wav2vec2-commonvoice-en']
                                    }
class asr_data_info():
    def __init__(self):
        self.test_datasets = ['librispeech-clean', 'librispeech-other', 'commonvoice-clean', 'wspire']


ASR_MODEL_INFO  = asr_model_info()
ASR_DATA_INFO = asr_data_info()


def transcribe(path, modelname):
    asr_model = EncoderDecoderASR.from_hparams(source=ASR_MODEL_INFO.model_dependencies[modelname][0],
                                            savedir=ASR_MODEL_INFO.model_dependencies[modelname][1])
    return asr_model.transcribe_file(path)
