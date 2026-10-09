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
    --epochs 300 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 4 \
    --lr 0.00004 \
    --no-balanced-sampler \
    --output-dir result/RAF-DB \
    > result/RAF-DB/train.log 2>&1 </dev/null &
```

Monitor the run with:

```bash
tail -f result/RAF-DB/train.log
```

The script expects seven class folders in both splits and saves
`best_model.pth`, `metrics.csv`, `metrics.json`, and `class_names.json`.
Because this folder layout has no separate test split, the reported metric is
validation accuracy.

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
    --epochs 300 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 4 \
    --lr 0.00004 \
    --no-balanced-sampler \
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


