from beautifultable import BeautifulTable

class tts_data_info():
    def __init__(self):
        self.test_datasets = {'ljspeech':'https://keithito.com/LJ-Speech-Dataset/'}
        self.test_speakers = {'ljspeech': '1'}
        self.test_sentences = {'ljspeech': '13100'}
        self.test_durations = {'ljspeech': '24 hours'}
        self.test_datasets_ = {'ljspeech_sentences': '500', 
                                'fastspeech_hard_sentences': '42', 
                               'common_voice_sentences': '50'}
        self.test_datasets_source = {'ljspeech_sentences': 'All sentences from LJSpeech test set from tacotron2 training', 
                                    'fastspeech_hard_sentences': 'Sentences which are known to peform poorly in TTS, manually listed in FastSpeech paper', 
                                    'common_voice_sentences': 'Sentences randomly selected from common voice corpus'}
        self.test_datasets_eval = {'ljspeech_sentences': 'MOS and MCD', 
                                'fastspeech_hard_sentences': 'MOS', 
                               'common_voice_sentences': 'MOS'}
                            
TTS_DATA_INFO = tts_data_info()

def list_datasets():
    table = BeautifulTable()
    table.column_headers = ["dataset", "num speakers","num sentences","duration", "link"]
    print('Training')
    for key in TTS_DATA_INFO.test_datasets:
        table.append_row([key,
                        TTS_DATA_INFO.test_speakers[key],
                        TTS_DATA_INFO.test_sentences[key],
                        TTS_DATA_INFO.test_durations[key],
                        TTS_DATA_INFO.test_datasets[key]
                        ])
    print(table)
    print('Evaluation')
    table = BeautifulTable()
    table.column_headers = ["dataset","num sentences","source", "evaluation metrics"]
    for key in TTS_DATA_INFO.test_datasets_:
        table.append_row([key,
                        TTS_DATA_INFO.test_datasets_[key],
                        TTS_DATA_INFO.test_datasets_source[key],
                        TTS_DATA_INFO.test_datasets_eval[key],
                        ])
    print(table)
    print('MOS: Mean Opinion Score (subjective evaluation)')
    print('MCD: Mel Cepstral Distortion (objective evaluation)')
TTS_DATA_INFO = tts_data_info()

def list_datasets():
    table = BeautifulTable()
    table.column_headers = ["dataset", "num speakers","num sentences","duration", "link"]
    for key in TTS_DATA_INFO.test_datasets:
        table.append_row([key,
                        TTS_DATA_INFO.test_speakers[key],
                        TTS_DATA_INFO.test_sentences[key],
                        TTS_DATA_INFO.test_durations[key],
                        TTS_DATA_INFO.test_datasets[key]
                        ])
    print(table)
 
class tts_model_info():
    def __init__(self):
        self.tts_models = ['tacotron2', 'fastspeech', 'glow-tts']
        self.vocoders = ['WaveGlow', 'HiFiGAN']

        self.model_paper = {'tacotron2':'https://arxiv.org/pdf/1712.05884.pdf',
                            'fastspeech':'https://arxiv.org/pdf/1905.09263.pdf',
                            'glow-tts':'https://arxiv.org/pdf/2005.11129.pdf',
                            'WaveGlow':'https://arxiv.org/pdf/1811.00002.pdf',
                            'HiFiGAN':'https://arxiv.org/pdf/2010.05646.pdf'}

        self.repository = {'tacotron2':'https://github.com/NVIDIA/tacotron2',
                            'fastspeech':'https://github.com/xcmyz/FastSpeech',
                            'glow-tts':'https://github.com/jaywalnut310/glow-tts',
                           'WaveGlow':'https://github.com/NVIDIA/waveglow',
                           'HiFiGAN':'https://github.com/jik876/hifi-gan'}

        self.pretrained_dataset = {'tacotron2':'LJSpeech',
                                'fastspeech':'LJSpeech',
                                'glow-tts':'LJSpeech',
                                'WaveGlow':'LJSpeech',
                                'HiFiGAN':'LJSpeech'}
        

TTS_MODEL_INFO = tts_model_info()

def list_models():
    table = BeautifulTable()
    table.column_headers = ["model name", "publication","code","pretrained"]
    print('Text to Spectogram')
    for key in TTS_MODEL_INFO.tts_models:
        table.append_row([key,
                        TTS_MODEL_INFO.model_paper[key],
                        TTS_MODEL_INFO.repository[key],
                        TTS_MODEL_INFO.pretrained_dataset[key],
                        ])
        
        
    print(table)
    print('Vocoder')
    table = BeautifulTable()
    table.column_headers = ["model name", "publication","code","pretrained"]
    for key in TTS_MODEL_INFO.vocoders:
        table.append_row([key,
                        TTS_MODEL_INFO.model_paper[key],
                        TTS_MODEL_INFO.repository[key],
                        TTS_MODEL_INFO.pretrained_dataset[key],
                        ])
    print(table)
    
def extract_test_pairs(dataset_name, dataset_dict, ljspeech_path):
    with open(dataset_dict[dataset_name], 'r') as f:
        data = f.read()
        
    if dataset_name == 'ljspeech_sentences':
        data = data.split('\n')[:-1]
        test_keys = {os.path.join(ljspeech_path, k.split('|')[0].split('/')[-1]):k.split('|')[-1] for k in data} 
    
    elif dataset_name == 'fastspeech_hard_sentences':
        test_keys = {d.split('.')[0]:'.'.join(d.split('.')[1:]).strip() for d in data.split('\n') if len(d)>0}

    elif dataset_name == 'common_voice_sentences':
        test_keys = {f'{i}':d for i, d in enumerate(data.split('\n')) if len(d)>0}

    else:
        raise NotImplementedError
    
    return test_keys
