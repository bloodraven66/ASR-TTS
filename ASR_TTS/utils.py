from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import soundfile as sf
from pathlib import Path
import re

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
    audio_files = get_files(path, '.flac')
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

import re

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
