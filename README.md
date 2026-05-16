# [ICASSP2026] T-Cache: Fast Inference for Masked Generative Transformer-Based TTS via Prompt-Aware Feature Caching

This folder contains the T-Cache-enabled MaskGCT inference files. The code is intended to be used on top of the original MaskGCT implementation from Amphion.

T-Cache adds feature caching during inference while keeping the original MaskGCT model structure and checkpoints compatible.

## 1. Install the Original MaskGCT Environment

First, install MaskGCT by following the official Amphion MaskGCT inference instructions:

- Clone the Amphion repository.
- Install the dependencies required by `models/tts/maskgct/requirements.txt`.
- Download or allow the scripts to download the original MaskGCT checkpoints.
- Verify that the original MaskGCT inference script runs correctly before applying the T-Cache files.

This repository does not replace the full MaskGCT installation guide. It only provides the files needed to run the T-Cache version after the original MaskGCT environment is working.

## 2. Replace the MaskGCT Files

After the original MaskGCT environment is installed, replace the following files in the original `models/tts/maskgct/` folder with the T-Cache versions from this folder:

```text
models/tts/maskgct/maskgct_s2a.py
models/tts/maskgct/maskgct_t2s.py
models/tts/maskgct/maskgct_utils.py
models/tts/maskgct/llama_nar.py
```

These files contain the inference-time changes required for T-Cache.

## 3. Add the T-Cache Support Files

In addition to replacing the files above, add the following T-Cache files and folders to `models/tts/maskgct/`:

```text
models/tts/maskgct/caching_conf.py
models/tts/maskgct/config_tcache/
models/tts/maskgct/samples/

```

The `caching_conf.py` file initializes the cache state and scheduling logic used during inference.

The `config_tcache/` folder contains the configuration files for baseline and T-Cache inference:

```text
config_tcache/config_baseline.json
config_tcache/config_tcache.json
```

## 4. Attention Implementation

T-Cache modifies the attention forward pass used during inference.

To avoid directly editing the original Hugging Face implementation, this repository uses monkey patching. The patched attention forward implementation is defined in:

```text
models/tts/maskgct/llama_nar.py
```

This keeps the original Hugging Face source code untouched while allowing the MaskGCT model to use the T-Cache attention behavior at runtime.

## 5. Run Inference

Once the files are replaced and the T-Cache support files are added, run inference from the Amphion project root.

Example:

```bash
python -m models.tts.maskgct.maskgct_inference seedtts_en config_tcache.json
```

For baseline inference without T-Cache, use:

```bash
python -m models.tts.maskgct.maskgct_inference seedtts_en config_baseline.json
```

The inference script currently supports the sample dataset options configured inside `maskgct_inference.py`, such as `seedtts_en` and `libri`.

## Notes

- Follow the original MaskGCT installation steps before using these files.
- The original MaskGCT checkpoints remain compatible.
- The Hugging Face attention implementation is not modified directly.
- The T-Cache behavior is controlled through the provided configuration files.

## Acknowledgments

We thank the authors of the following projects for releasing the implementations and ideas that this work builds upon:

- [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
- [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)
- [ToCa](https://github.com/Shenyi-Z/ToCa)
- [FORA](https://github.com/prathebaselva/FORA)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{irihose2026t,
  title={T-Cache: Fast Inference For Masked Generative Transformer-Based TTS Via Prompt-Aware Feature Caching},
  author={Irihose, Obed and Zhang, Le},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={16702--16706},
  year={2026},
  organization={IEEE}
}
```
## Contact
For questions or issues, please contact:
```text
Email: 2672291403ATqq.com
```
