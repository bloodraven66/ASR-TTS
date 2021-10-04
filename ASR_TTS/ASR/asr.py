from speechbrain.pretrained import EncoderDecoderASR

def transcribe(path):
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
    asr_model.transcribe_file(path)
