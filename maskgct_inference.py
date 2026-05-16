# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Note: this script is hardcoded for quick inference tests on a
# small set of sample files. If you want to run a different dataset, add its
# paths in the dataset selection block inside main().

from models.tts.maskgct.maskgct_utils import *
from models.tts.maskgct.caching_conf import read_wav_text_files
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf

from pathlib import Path
import os
from time import time
import argparse

BASE_DIR = Path(__file__).resolve().parent


def main(dataset, inference_config_path, outdir_path="./output", device="cuda:0"):
    if outdir_path is None:
        outdir_path = "./output"

    assert dataset is not None, "dataset should be provided"
    assert inference_config_path is not None, "inference_config_path should be provided"

    ###### HARDCODED TEST FILES ############################

    if dataset == "seedtts_en":
        dataset_path= BASE_DIR /'../../../Samples/audios/Seed_tts_en/'
        wav_txt_path= BASE_DIR / '../../../Samples/filelists/seedtts_en.lst'


    elif dataset == "libri":
        dataset_path= BASE_DIR /'../../../Samples/audios/LibriSpeech/test-clean'
        wav_txt_path= BASE_DIR / '../../../Samples/filelists/libri_speech.lst'

    else:
        raise ValueError(
            "So far, we are only running a small test using a few samples from "
            "seedtts_en and LibriSpeech test-clean. Please add your custom dataset."
        )
    
    print(f"Using dataset: {dataset}")
    print(f"Dataset path: {dataset_path}")
    print(f"File list: {wav_txt_path}")


    lang='en'
    dataset_path = dataset_path.resolve()
    wav_txt_path = wav_txt_path.resolve()

    # build model
    device = torch.device(device)
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    inference_config = load_config(os.path.join("./models/tts/maskgct/config_tcache",inference_config_path))

    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)


    #download checkpoint
    #download semantic codec ckpt
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    # download acoustic codec ckpt
    codec_encoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
    )
    codec_decoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
    )
    # download t2s model ckpt
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    )
    # download s2a model ckpt
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors"
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors"
    )


    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    # inference

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )
    
    target_wavs, prompt_texts, prompt_wavs, target_texts = read_wav_text_files(
        wav_txt_path,
        dataset_path,
    )

    samples_time = []
    samples = list(zip(target_wavs, prompt_texts, prompt_wavs, target_texts))
    samples_warmup = samples[:5]

    # Run a short warmup first to initialize CUDA kernels.
    for target_wav, prompt_text, prompt_wav, target_text in tqdm(
        samples_warmup,
        desc="Warmup in progress...",
    ):
        start_time = time()

        try:
            maskgct_inference_pipeline.maskgct_inference(
                prompt_wav,
                prompt_text,
                target_text,
                language=lang,
                target_language=lang,
                kwargs_t2s=inference_config.t2s,
                kwargs_s2a=inference_config.s2a,
                n_timesteps=inference_config.t2s.n_steps_t2s,
                n_timesteps_s2a=inference_config.s2a.n_steps_s2a,
            )

            elapsed_time = time() - start_time
            samples_time.append([target_wav, elapsed_time])

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Skipping to the next sample.")
            continue

        except Exception as e:
            raise RuntimeError(f"Error occurred during warmup for {target_wav}") from e

    mean_time = sum(item[1] for item in samples_time) / len(samples_time)

    print("=================================RESULTS======================================================")
    print(f" The average time used for Warmup {len(samples_time)} samples is {mean_time} seconds/sample")
    print("==============================================================================================")

    samples_time = []  


    for target_wav, prompt_text, prompt_wav, target_text in tqdm(
        samples,
        total=len(prompt_wavs),
        desc="Inference in progress...",
    ):
        start_time = time()

        try:
            maskgct_inference_pipeline.maskgct_inference(
                prompt_wav,
                prompt_text,
                target_text,
                language=lang,
                target_language=lang,
                kwargs_t2s=inference_config.t2s,
                kwargs_s2a=inference_config.s2a,
                n_timesteps=inference_config.t2s.n_steps_t2s,
                n_timesteps_s2a=inference_config.s2a.n_steps_s2a,
            )

            elapsed_time = time() - start_time
            samples_time.append([target_wav, elapsed_time])

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Skipping to the next sample.")
            continue

        except Exception as e:
            raise RuntimeError(f"Error occurred during warmup for {target_wav}") from e

    mean_time = sum(item[1] for item in samples_time) / len(samples_time)

    print("=================================RESULTS==============================================")
    print(f" The average time used for {len(samples_time)} samples is {mean_time} seconds/sample")
    print("======================================================================================")

    return samples_time


def parse_args():
    parser = argparse.ArgumentParser(description="Run MaskGCT inference.")
    parser.add_argument("dataset", choices=["seedtts_en", "libri"])
    parser.add_argument("inference_config_path")
    parser.add_argument("outdir_path", nargs="?", default="./output")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset=args.dataset,
        inference_config_path=args.inference_config_path,
        outdir_path=args.outdir_path,
        device=args.device,
    )



