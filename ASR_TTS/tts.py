from beautifultable import BeautifulTable

class tts_data_info():
    def __init__(self):
        self.test_datasets = {'ljspeech':'https://keithito.com/LJ-Speech-Dataset/'}
        self.test_speakers = {'ljspeech': '1'}
        self.test_sentences = {'ljspeech': '13100'}
        self.test_durations = {'ljspeech': '24 hours'}
        
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