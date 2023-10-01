# pydx7
A Python Implementation of the Yamaha DX7 Envelope Generator and 6-op synthesizer FM Engine.

This package is used to load DX7 patches from a sysex file and render the envelopes of its 6 oscillators.
This was developed as a data generator for the [fmtransfer](https://github.com/fcaspe/fmtransfer) paper.

## Installation

Clone this repository and run
```bash
pip install -e .
pip install -r requirements.txt
```

## Acknowledgements
 - The DX7 Envelope Generator implementation was adapted from [Dexed](https://github.com/asb2m10/dexed)
 - The patch unpacking routine was adapted from [learnfm](https://github.com/bwhitman/learnfm)

## Citation
If you find this work useful, please consider citing us:

```bibtex
@article{caspe2023learnedenvelopes,
    title={{FM Tone Transfer with Learned Envelopes}},
    author={Caspe, Franco and McPherson, Andrew and Sandler, Mark},
    journal={Proceedings of Audio Mostly 2023},
    year={2023}
}
```
