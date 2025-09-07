#!flask/bin/python
import argparse
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs

from flask import Flask, render_template, render_template_string, request, send_file

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)

use_multi_language = hasattr(synthesizer.tts_model, "num_languages") and (
    synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
)
language_manager = getattr(synthesizer.tts_model, "language_manager", None)

# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    # 获取模型信息 - 优先检查config路径来识别XTTS模型
    model_name = args.model_name if args.model_name else "custom"
    
    # 如果使用了自定义模型路径，通过config路径判断是否是XTTS
    if args.model_path and args.config_path:
        is_xtts = "xtts" in args.config_path.lower()
        is_vctk = False
        is_ljspeech = False
    else:
        # 使用预训练模型名称判断
        is_xtts = "xtts" in model_name.lower() if model_name else False
        is_vctk = "vctk" in model_name.lower() if model_name else False
        is_ljspeech = "ljspeech" in model_name.lower() if model_name else False
    
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=use_multi_speaker,
        use_multi_language=use_multi_language,
        speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None,
        language_ids=language_manager.name_to_id if language_manager is not None else None,
        use_gst=use_gst,
        model_name=model_name,
        is_xtts=is_xtts,
        is_vctk=is_vctk,
        is_ljspeech=is_ljspeech,
    )


@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)

        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        print(f" > use_multi_speaker: {use_multi_speaker}")
        print(f" > use_multi_language: {use_multi_language}")
        
        # 根据模型类型决定如何调用
        # 检查是否是XTTS模型（通过config路径或者模型名称）
        is_xtts_model = False
        if args.model_path and args.config_path:
            is_xtts_model = "xtts" in args.config_path.lower()
        elif args.model_name:
            is_xtts_model = "xtts" in args.model_name.lower()
            
        if is_xtts_model:
            # XTTS模型 - 需要speaker和language参数（即使为空）
            wavs = synthesizer.tts(text, speaker_name=speaker_idx, language_name=language_idx, style_wav=style_wav)
        else:
            # 其他模型 - 动态构建参数
            tts_kwargs = {"text": text}
            
            # 只在多说话人模型且有speaker_idx时传递speaker参数
            if use_multi_speaker and speaker_idx:
                tts_kwargs["speaker_name"] = speaker_idx
                
            # 只在多语言模型且有language_idx时传递language参数
            if use_multi_language and language_idx:
                tts_kwargs["language_name"] = language_idx
                
            # style_wav可以总是传递（如果不为None）
            if style_wav:
                tts_kwargs["style_wav"] = style_wav
            
            wavs = synthesizer.tts(**tts_kwargs)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


@app.route("/api/tts_with_clone", methods=["POST"])
def tts_with_clone():
    """TTS with voice cloning from uploaded audio file"""
    with lock:
        text = request.form.get("text", "")
        language_idx = request.form.get("language_id", "en")
        
        # Get uploaded file
        speaker_wav_file = request.files.get("speaker_wav")
        
        if not text:
            return "No text provided", 400
            
        if not speaker_wav_file:
            return "No audio file provided", 400
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            speaker_wav_file.save(tmp_file.name)
            temp_filename = tmp_file.name
        
        try:
            print(f" > Model input: {text}")
            print(f" > Language: {language_idx}")
            print(f" > Voice clone from: {speaker_wav_file.filename}")
            
            # Use the uploaded audio file for voice cloning
            wavs = synthesizer.tts(
                text=text, 
                language_name=language_idx,
                speaker_wav=temp_filename,  # Path to the uploaded audio file
                speaker_name=None  # Don't use preset speaker when cloning
            )
            
            out = io.BytesIO()
            synthesizer.save_wav(wavs, out)
            
            return send_file(out, mimetype="audio/wav")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        print(f" > Model input: {text}")
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()
