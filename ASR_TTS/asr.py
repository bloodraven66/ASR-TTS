from .utils import data_parse_libri, find_cer
from tqdm import tqdm
# import torchaudio
# from torch.nn.utils.rnn import pad_sequence
# import torch
from beautifultable import BeautifulTable


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
        self.asr = {'rnn': 'RNN',
                    'transformers': 'Transformers',
                    'wav2vec': 'Wav2vec'
                        }
        self.lm = {'rnn': 'RNN',
                    'transformers': 'Transformers',
                    'wav2vec': 'Transformers'
                        }
        self.pretrained = {'rnn': 'https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech',
                    'transformers': 'https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech',
                    'wav2vec': ' https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en'
                        }
        self.pretrained_dataset = {'rnn': 'LibriSpeech',
                    'transformers': 'LibriSpeech',
                    'wav2vec': 'Common-voice'
                        }

class asr_data_info():
    def __init__(self):
        self.test_datasets = {
                            'librispeech-clean':'https://www.openslr.org/12',
                            'librispeech-other':'https://www.openslr.org/12',
                            'commonvoice-clean':'https://commonvoice.mozilla.org/en/datasets',
                            'whisper-spire':'-'
                            }
        self.test_speakers = {
                            'librispeech-clean':'40',
                            'librispeech-other':'33',
                            'commonvoice-clean':'3995',
                            'whisper-spire':'88'
                            }
        self.test_sentences = {
                            'librispeech-clean':'2620',
                            'librispeech-other':'2939',
                            'commonvoice-clean':'3995',
                            'whisper-spire':'4396'
                            }
        self.test_durations = {
                            'librispeech-clean':'5.4 hours',
                            'librispeech-other':'5.1 hours',
                            'commonvoice-clean':'4.9 hours',
                            'whisper-spire':'3.8 hours'
                            }


ASR_MODEL_INFO  = asr_model_info()
ASR_DATA_INFO = asr_data_info()

def select_model(name):
    assert name in ASR_MODEL_INFO.models
    return ASR_MODEL_INFO.model_dependencies[name][0], ASR_MODEL_INFO.model_dependencies[name][1]

def list_datasets():
    table = BeautifulTable()
    table.column_headers = ["dataset", "num speakers","num sentences","duration", "link"]
    for key in ASR_DATA_INFO.test_datasets:
        table.append_row([key,
                        ASR_DATA_INFO.test_speakers[key],
                        ASR_DATA_INFO.test_sentences[key],
                        ASR_DATA_INFO.test_durations[key],
                        ASR_DATA_INFO.test_datasets[key]
                        ])
    print(table)
# list_datasets()

def list_models():
    table = BeautifulTable()
    table.column_headers = ["model name", "ASR","LM","pretrained", "link"]
    for key in ASR_MODEL_INFO.models:
        table.append_row([key,
                        ASR_MODEL_INFO.asr[key],
                        ASR_MODEL_INFO.lm[key],
                        ASR_MODEL_INFO.pretrained_dataset[key],
                        ASR_MODEL_INFO.pretrained[key]
                        ])
    print(table)

def libri_transcribe(path, model, limit_sentence_count=None, batch_size=2, use_batch=True):
    path = data_parse_libri(path)
    cer = []
    wer = []
    audio_files, text = [], []
    cers, wers = [], []
    if use_batch:
        asr_model = transcribe(None, model, True, True)
    for key in tqdm(path):
        if use_batch:
            if len(audio_files)<batch_size:
                audio_files.append(key)
                text.append(path[key])
                if len(cer)>0:
                    if len(audio_files)+len(cer)>=limit_sentence_count:
                        break
            else:
                sigs=[]
                lens=[]
                for audio_file in audio_files:
                    snt, fs = torchaudio.load(audio_file)
                    sigs.append(snt.squeeze())
                    lens.append(snt.shape[1])
                batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
                lens = torch.Tensor(lens) / batch.shape[1]
                preds = asr_model.transcribe_batch(batch, lens)[0]
                for i in range(len(audio_files)):
                    cer_val, wer_val, cer_, wer_ = find_cer(preds[i], text[i])
                    cers.append(cer_val)
                    wers.append(wer_val)
                cer.extend(cers)
                wer.extend(wers)

                audio_files, text = [], []
                cers, wers = [], []
        else:
            pred = transcribe(key, model, True)
            cer_val, wer_val, cer_, wer_ = find_cer(pred, path[key])
            cer.append(cer_val)
            wer.append(wer_val)
            if limit_sentence_count is not None:
                if len(cer)>=limit_sentence_count:
                    break

    print(f'CER: {sum(cer)/len(cer)}, WER: {sum(wer)/len(wer)}')
# libri_transcribe('../../../other_tts_data/librispeech/test_clean/test_clean/', 'wav2vec', limit_sentence_count=5)
