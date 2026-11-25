# POSTER
The project is an official implementation of our paper [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://arxiv.org/pdf/2204.04083.pdf).


### Preparation
- create conda environment (we provide requirements.txt)

- Data Preparation

  Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset, and make sure it have a structure like following:
 
	```
	- data/raf-basic/
		 EmoLabel/
		     list_patition_label.txt
		 Image/aligned/
		     train_00001_aligned.jpg
		     test_0001_aligned.jpg
		     ...
	```

- Pretrained model weights
Dowonload pretrain weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Put entire `pretrain` folder under `models` folder.

	```
	- models/pretrain/
		 ir50.pth
		 mobilefacenet_model_best.pth.tar
		     ...
	```

### Testing

Our best model can be download from [here](https://drive.google.com/drive/folders/1jeCPTGjBL8YgKKB9YrI9TYZywme8gymv?usp=sharing), put under `checkpoint ` folder. You can evaluate our model on RAD-DB dataset by running: 

```
python test.py --checkpoint checkpoint/rafdb_best.pth -p
```

### Training
Train on RAF-DB dataset:
```
python train.py --gpu 0,1 --batch_size 200
```
You may adjust batch_size based on your # of GPUs. Usually bigger batch size can get higher performance. We provide the log in  `log` folder. You may run several times to get the best results. 

### Server RAF-DB folder format

The server dataset can be used directly with the following layout:

```
/mnt/data/yanyi2025/cyj/raf-db/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
```

The folder-based trainer uses POSTER's model, SAM optimizer, label smoothing,
and optional inverse-frequency sampling. It uses only physical GPU 0 and
saves the best validation checkpoint under `poster/result/RAF-DB`.

First make sure the official backbone files are present:

```
models/pretrain/ir50.pth
models/pretrain/mobilefacenet_model_best.pth.tar
```

Then run from the repository root:

```bash
mkdir -p result/RAF-DB

nohup env CUDA_VISIBLE_DEVICES=0 python train_raf_folder.py \
    --data-root /mnt/data/yanyi2025/cyj/raf-db \
    --gpu 0 \
    --modeltype large \
    --head-type linear \
    --epochs 300 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 4 \
    --lr 0.00004 \
    --no-balanced-sampler \
    --amp \
    --tta \
    --output-dir result/RAF-DB \
    > result/RAF-DB/train.log 2>&1 </dev/null &
```

Monitor the run with:

```bash
tail -f result/RAF-DB/train.log
```

`--amp` uses BF16 autocast when the GPU supports it and otherwise falls back
to FP32. `--tta` averages the original and horizontally flipped images during
validation. The trainer writes `latest_model.pth` after every completed epoch;
resume an interrupted run with:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python -u train_raf_folder.py \
    --data-root /mnt/data/yanyi2025/cyj/raf-db \
    --gpu 0 \
    --modeltype large \
    --head-type linear \
    --epochs 300 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 4 \
    --lr 0.00004 \
    --no-balanced-sampler \
    --amp \
    --tta \
    --resume result/RAF-DB/latest_model.pth \
    --output-dir result/RAF-DB \
    > result/RAF-DB/resume.log 2>&1 </dev/null &
```

The script expects seven class folders in both splits and saves
`best_model.pth`, `metrics.csv`, `metrics.json`, and `class_names.json`.
Because this folder layout has no separate test split, the reported metric is
validation accuracy.

The default configuration keeps the pretrained landmark backbone frozen.
Comparisons with experimental variants should use the same batch size, AMP,
TTA, seed, and data split.

The experimental local-token readout can be enabled with
`--readout-type local`. It uses attention pooling over final facial and
landmark tokens, then adds the result through a zero-initialized residual
scale so the original CLS readout is preserved at initialization.

An OTP-NS-style sample-aware loss is available with `--loss-type otpns`. It
uses an EMA teacher to estimate per-sample confidence, reduce the weight of
uncertain samples, and dynamically smooth their targets. This is an
implementation inspired by OTP-NS rather than a full teacher-pool reproduction.

### DiDiCM-inspired classification head

Use `--readout-type local --head-type didicm --spatial-enhancement none` in
either folder trainer to keep the POSTER backbone and local-token readout while
refining the class posterior with a compact learned denoising head. This is a
lightweight DiDiCM-inspired adaptation, not a reproduction of the original
CVPR 2026 method or its full discrete diffusion training objective. Keep the
same optimizer, loss, data splits, batch size, AMP, TTA and seed as the
local-attention baseline, and use a fresh output directory.

### Local attention with heat-conduction enhancement

Use `--readout-type local --head-type linear --spatial-enhancement hco` to
apply a DCT-based heat-conduction operator to POSTER's 7x7 fused image tokens.
The adapter operates after pyramid fusion and normalization but BEFORE local
attention pooling. Only the 49 image tokens are enhanced; CLS and landmark
tokens retain their existing paths. Local readout is required for this mode.

The reference is `Heat2D` in the official vHeat repository:
https://github.com/MzeroMiko/vHeat/blob/main/classification/models/vHeat.py
The core is orthonormal DCT-II, per-frequency/per-channel heat decay
`exp(-k * ((pi*u/H)^2 + (pi*v/W)^2))`, inverse DCT, output normalization,
SiLU gating, and projection. Learnable frequency embeddings predict `k`.
POSTER-specific choices are a pre-normalization, softplus rather than ReLU
for positive diffusivity, FP32 spectral calculations under AMP, and a
per-channel residual scale initialized to 1e-3. No extra FFN is added.

This is an HCO adaptation, not a full vHeat backbone reproduction. The existing
zero-initialized local-readout scale is unchanged; HCO starts receiving
classification gradients once that readout becomes active. Shared baseline
parameters retain their seeded initialization. The optimizer, loss and
inference protocol are unchanged. Use a fresh directory, not a DiDiCM or
multi-scale checkpoint, and compare with local-attention alone. Accuracy
improvements are not guaranteed.

RAF-DB (physical GPU 1, batch size 128; run from the server repository):

```bash
OUT=/mnt/data/yanyi2025/cyj/otherModels/poster/result/RAF-DB/local-attention-hco
mkdir -p "$OUT"
nohup python -u train_raf_folder.py --data-root /mnt/data/yanyi2025/cyj/raf-db --gpu 1 --modeltype large --head-type linear --readout-type local --spatial-enhancement hco --loss-type baseline --epochs 300 --batch-size 128 --val-batch-size 32 --workers 4 --lr 0.00004 --no-balanced-sampler --amp --tta --seed 123 --output-dir "$OUT" > "$OUT/train.log" 2>&1 < /dev/null &
tail -f "$OUT/train.log"
```

FERPlus (run separately, or select a different GPU for concurrent training):

```bash
OUT=/mnt/data/yanyi2025/cyj/otherModels/poster/result/FERPLUS/local-attention-hco
mkdir -p "$OUT"
nohup python -u train_fer_folder.py --data-root /mnt/data/yanyi2025/cyj/ferplus/fer_img --gpu 1 --modeltype large --head-type linear --readout-type local --spatial-enhancement hco --loss-type baseline --epochs 300 --batch-size 128 --val-batch-size 32 --workers 4 --lr 0.00004 --no-balanced-sampler --amp --tta --seed 123 --output-dir "$OUT" > "$OUT/train.log" 2>&1 < /dev/null &
tail -f "$OUT/train.log"
```

The earlier local-attention multi-scale spatial experiment remains selectable
with `--spatial-enhancement multiscale` for controlled comparisons.

### Local attention with multi-scale spatial enhancement (previous experiment)

Use `--readout-type local --spatial-enhancement multiscale --head-type linear`
in either folder trainer. This replaces the previous part-prototype experiment.
The optional adaptor runs after `pyramid_fuse` over its 7x7 image tokens. Three
depthwise convolution branches use dilation 1, 2, and 3, followed by gated
multi-scale fusion and a zero-initialized residual projection. This keeps the
POSTER backbone and classifier unchanged and starts close to the local-attention
model. It is inspired by lightweight visual adaptors; it is not a reproduction
of Mamba-Adaptor and does not implement an actual state-space/Mamba layer.

Set batch size to 128 and match AMP, TTA, learning rate, seed and data splits to
the local-attention baseline. Use a fresh output directory and do not resume a
part-prototype checkpoint. The adaptor is enabled at inference and its weights
are saved in the normal model checkpoint.

Suggested output folders: `result/RAF-DB/local-attention-multiscale` and
`result/FERPLUS/local-attention-multiscale`. Accuracy improvements are not
guaranteed; compare against local-attention alone.

Run the focused tests without pretrained weight files:

```bash
python -m unittest discover -s tests -v
```

### Server FER2013 folder format

The FER2013 image dataset can be used directly with the following layout:

```
/mnt/data/yanyi2025/cyj/fer2013_img/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg
```

Run 300 epochs on physical GPU 0 from the repository root:

```bash
mkdir -p result/fer2013

nohup env CUDA_VISIBLE_DEVICES=0 python train_fer_folder.py \
    --data-root /mnt/data/yanyi2025/cyj/fer2013_img \
    --gpu 0 \
    --modeltype large \
    --head-type linear \
    --epochs 300 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 4 \
    --lr 0.00004 \
    --no-balanced-sampler \
    --amp \
    --tta \
    --output-dir result/fer2013 \
    > result/fer2013/train.log 2>&1 </dev/null &
```

Monitor the training log with:

```bash
tail -f result/fer2013/train.log
```

The FER2013 trainer selects `best_model.pth` using validation accuracy and
evaluates that checkpoint on the test split. It saves `metrics.csv`,
`metrics.json`, and `class_names.json` alongside the checkpoint.


## License

Our research code is released under the MIT license. See [LICENSE](LICENSE) for details. 



## Citations
If you find our work useful in your research, please consider citing:

```bibtex
@article{zheng2022poster,
  title={Poster: A pyramid cross-fusion transformer network for facial expression recognition},
  author={Zheng, Ce and Mendieta, Matias and Chen, Chen},
  journal={arXiv preprint arXiv:2204.04083},
  year={2022}
}
```


## Acknowledgments

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[JiaweiShiCV/Amend-Representation-Module](https://github.com/JiaweiShiCV/Amend-Representation-Module) 


