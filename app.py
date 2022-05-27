import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gradio as gr
import os


api_token = os.getenv("API_TOKEN")
model_name = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=api_token)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=api_token)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def convert(inputfile, outfile):
    target_sr = 16000
    data, sample_rate = librosa.load(inputfile)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(outfile, data, target_sr)


def parse_transcription(wav_file):
    filename = wav_file.name.split('.')[0]
    convert(wav_file.name, filename + "16k.wav")
    speech, _ = sf.read(filename + "16k.wav")
    input_values = processor(speech, sampling_rate=16_000, return_tensors="pt").input_values
    input_values = input_values.to(device)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription


output = gr.outputs.Textbox(label="The transcript")

input_ = gr.inputs.Audio(source="microphone", type="file")

gr.Interface(parse_transcription, inputs=input_,  outputs=[output],
             analytics_enabled=False,
             title="Multilingual Speech Recognition for Indonesian Languages",
             description="Automatic Speech Recognition Live Demo for Indonesian, Javanese and Sundanese Language",
             article="This demo was built for the project "
                     "<a href='https://github.com/indonesian-nlp/multilingual-asr' target='_blank'>Multilingual Speech Recognition for Indonesian Languages</a>. "
                     "It uses the <a href='https://huggingface.co/indonesian-nlp/wav2vec2-indonesian-javanese-sundanese' target='_blank'>indonesian-nlp/wav2vec2-indonesian-javanese-sundanese</a> model "
                     "which was fine-tuned on Indonesian Common Voice, Javanese and Sundanese OpenSLR speech datasets."
             ).launch( inline=False, server_name="0.0.0.0", show_tips=False, enable_queue=True)
