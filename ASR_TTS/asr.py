from .utils import data_parse_libri, find_cer
from tqdm import tqdm
# import torchaudio
# from torch.nn.utils.rnn import pad_sequence
# import torch

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

def select_model(name):
    assert name in ASR_MODEL_INFO.models
    return ASR_MODEL_INFO.model_dependencies[name][0], ASR_MODEL_INFO.model_dependencies[name][1]


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
