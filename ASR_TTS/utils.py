from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import matplotlib.pylab as plt
import soundfile as sf
from pathlib import Path
import re
import os
import shutil

PATHS = {'librispeech-clean':'drive/MyDrive/ASR_datasets/test_sets/test-clean.tar.gz',
        'librispeech-other':'drive/MyDrive/ASR_datasets/test_sets/test-other.tar.gz',
        'whisper-spire': 'drive/MyDrive/ASR_datasets/test_sets/WSpire-test.zip',
        'commonvoice-clean':'drive/MyDrive/ASR_datasets/test_sets/cv-test.zip',
         }

NoDrive_PATHS = {'librispeech-clean':'ASR_datasets/test_sets/test-clean.tar.gz',
        'librispeech-other':'ASR_datasets/test_sets/test-other.tar.gz',
        'whisper-spire': 'ASR_datasets/test_sets/WSpire-test.zip',
        'commonvoice-clean':'ASR_datasets/test_sets/cv-test.zip',
         }

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saved!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""

def get_audio():
    """
    Record Audio from google colab notebook.
    Code taken from https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_colab/
    """
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    output, err = process.communicate(input=binary)
    riff_chunk_size = len(output) - 8
    q = riff_chunk_size
    b = []
    for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)
    riff = output[:4] + bytes(b) + output[8:]
    return wav_read(io.BytesIO(riff))

def save_audio(y, sr, path):
    print(f'Audio saved at {path}...')
    sf.write(path, y, sr)

def get_files(path, extension):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return [str(n) for n in list(path.rglob(f'*{extension}'))]

def data_parse_libri(path):
    audio_files = get_files(path, '.flac')download_from_google_drive
    files = get_files(path, '.txt')
    text_map = {}
    for filename in files:
        with open(filename, 'r') as f:
            data = f.read()
        data = data.split('\n')
        for line in data:
            if len(line)>0:
                textid = line.split('-')
                text_map['-'.join(textid[:2])+'-'+textid[2].split(' ')[0]] = ' '.join(' '.join(textid[2:]).split(' ')[1:])
    mapping = {}
    for filename in audio_files:
        id = filename.split('/')[-1].strip('.flac')
        text = text_map[id]
        mapping[filename] = text
    return mapping

def data_parse_cv(path):
    files = get_files(path, '.csv')
    audio_files = get_files(path, '.mp3')
    with open(files[0], 'r') as f:
        data = f.read()
    data = [data_ for data_ in data.split('\n') if len(data_)>0][1:]
    mapping = {os.path.join('/'.join(audio_files[0].split('/')[:-1]), data_.split(',')[0].split('/')[-1]): data_.split(',')[1] for data_ in data}
    return mapping

def data_parse_spire(path):
    audio_files = get_files(path, '.wav')
    files = get_files(path, '.txt')
    audio_maps = {}
    audio_name_maps = {}
    for filename in audio_files:
        key = Path(filename).stem[:-4].split('_')
        if '_'.join(key[:3]) not in audio_maps:
            audio_maps['_'.join(key[:3])] = [key[-1]]
            audio_name_maps['_'.join(key[:3])] = [filename]
        else:
            audio_name_maps['_'.join(key[:3])].append(filename)
            audio_maps['_'.join(key[:3])].append(key[-1])
    text_map = {}
    current_sub = set()
    for filename in files:
        with open(str(filename), 'r') as f:
            data = f.read()
        key = Path(filename).stem.split('_')[1:4]
        key = '_'.join(key)
        if key not in current_sub:
            current_sub.add(key)
            paths = [x for _, x in sorted(zip([int(i) for i in audio_maps[key]], audio_name_maps[key]))]
            keys = sorted([int(i) for i in audio_maps[key]])
        data = [data_ for data_ in data.split('\n') if len(data_)>2]
        key_text = {paths[idx]:data_.split('.')[1] for idx, data_ in enumerate(data)}
        text_map = {**text_map, **key_text}
    return text_map

def parse_files(path):
    key = path.split('/')[0]
    if key in ['librispeech-clean', 'librispeech-other']:
        return data_parse_libri(path)
    elif key == 'commonvoice-clean':
        return data_parse_cv(path)
    elif key == 'whisper-spire':
        return data_parse_spire(path)
    else:
        raise NotImplementedError

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]

def find_cer(sentence1, sentence2):
    wer = {}
    cer = {}
    keys = ['sub', 'ins', 'del', 'n']
    wer = {i: [] for i in keys}
    cer = {i: [] for i in keys}
    _, (s, i, d) = levenshtein(sentence1, sentence2)
    cer['sub'] = s
    cer['ins'] = i
    cer['del'] = d
    cer['n'] = len(sentence1)
    _, (s, i, d) = levenshtein(sentence1.split(), sentence2.split())
    wer['sub'] = s
    wer['ins'] = i
    wer['del'] = d
    wer['n'] = len(sentence1.split())
    cer_val = 100.0 * (cer['sub'] + cer['ins'] + cer['del']) / cer['n']
    wer_val = 100.0 * (wer['sub'] + wer['ins'] + wer['del']) / wer['n']
    return cer_val, wer_val, cer, wer

# data_stats(path='../../../other_tts_data/librispeech/test_other/test_other/')

def unpack_from_drive(key, from_drive=False):
    if from_drive:
        shutil.unpack_archive(PATHS[key], key)
    else:
        shutil.unpack_archive(NoDrive_PATHS[key], key)    
    return key

def plot_data_3x(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsizplot_data_3xe)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none', cmap='viridis')
        
def plot_data(data, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(data, aspect='auto', origin='bottom', 
                    interpolation='none', cmap='viridis')

# def download_pretrained_from_google_drive_(file_id, file_name):
#     !rm -f ./cookie
#     !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
#     confirm_text = !awk '/download/ {print $NF}' ./cookie
#     confirm_text = confirm_text[0]
#     !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm={confirm_text}&id={file_id}" -o {file_name}
    
def download_all_models(download_fn):
    models = {'tacotron2_checkpoint.pth':'1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA',
              'fastspeech_checkpoint.pth':'1vMrKtbjPj9u_o3Y-8prE6hHCc6Yj4Nqk',
              'glowtts_checkpoint.pth':'1JiCMBVTG4BMREK8cT3MYck1MgYvwASL0',
             'waveglow_checkpoint.pth':'1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx'}
    for modelname in models:
        print(f'Downloading pretrained model for {modelname[:-15]}, saving as {modelname}\n')
        download_fn(models[modelname], modelname)
        
    return list(models.keys())
