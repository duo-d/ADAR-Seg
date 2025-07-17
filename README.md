# 1. Installation
1. Download the source code with git

2. Create conda environment:
      ```
      conda create -y -n main python=3.9
      conda activate ADAR-Seg
      ```
3. Install pytorch 1.13.0
      ```
      pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
      ```
4. Install [Minkowski Engine v0.5.4](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#pip)

5. Install pytorch_lightning 1.9.0 with torchmetrics 1.4.0.post0
    ```
    pip install --no-cache-dir pytorch_lightning==1.9.0
    ```
   
7. Install the additional dependencies:
      ```
      cd main/
      pip install -r requirements.txt
      ```
7. Install `pytorch-scatter` with torch 1.13.0 and CUDA 11.7:
      ```
      pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
      ```

8. Install
      ```
      pip install -e ./
      ```
# 2. Data

Please download the following data into a folder e.g. `/gpfsdswork/dataset/SemanticKITTI` and unzip:

- **Semantic Scene Completion dataset v1.1** (`SemanticKITTI voxel data (700 MB)`) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download).

- **Point-wise semantic labels** (`SemanticKITTI label data (179 MB)`) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download).

- [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) calibration data (Download odometry data set `(calibration files, 1 MB)`). 

- [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) Velodyne data (Download odometry data set  `(velodyne laser data, 80 GB)`).

- The dataset folder at `/gpfsdswork/dataset/SemanticKITTI` should have the following structure:
    ```
    └── /gpfsdswork/dataset/SemanticKITTI
      └── dataset
        └── sequences
    ```

# 3. Panoptic labels generation
1. Create a folder to store preprocess data for Semantic KITTI dataset e.g. **/lustre/fsn1/projects/rech/kvd/uyl37fq/main_preprocess/kitti** .
2. Execute the command below to generate panoptic labels, or **move to the next step** to directly download the **pre-generated labels**:
      ```
      cd main/
      python label_gen/gen_instance_labels.py \
          --kitti_config=main/data/semantic_kitti/semantic-kitti.yaml \
          --kitti_root=/gpfsdswork/dataset/SemanticKITTI \
          --kitti_preprocess_root=/lustre/fsn1/projects/rech/kvd/uyl37fq/main_preprocess/kitti \
          --n_process=10
      ```
3. Your folder structure with the instance labels should look as follows:
      ```
      /lustre/fsn1/projects/rech/kvd/uyl37fq/main_preprocess/kitti360
      └── instance_labels_v2
          ├── 2013_05_28_drive_0000_sync
          ├── 2013_05_28_drive_0002_sync
          ├── 2013_05_28_drive_0003_sync
          ├── 2013_05_28_drive_0004_sync
          ├── 2013_05_28_drive_0005_sync
          ├── 2013_05_28_drive_0006_sync
          ├── 2013_05_28_drive_0007_sync
          ├── 2013_05_28_drive_0009_sync
          └── 2013_05_28_drive_0010_sync
      ```


### 4.1 Extract point features

> [!NOTE]
> This step is only necessary when training on SemanticKITTI because of the availability of the WaffleIron pretrained model.

> [!TIP]
> A better approach could be to explore the features of pretrained models available at [https://github.com/valeoai/ScaLR](https://github.com/valeoai/ScaLR).


1. Install WaffleIron in a separate conda environment:
      ```
      conda create -y -n waffleiron 
      conda activate waffleiron
      pip install pyYAML==6.0 tqdm==4.63.0 scipy==1.8.0 torch==1.11.0 tensorboard==2.8.0
      cd ADAR-Seg/WaffleIron_mod
      pip install -e ./
      ```

> [!CAUTION]
> I used the older version of WaffleIron which requires pytorch 1.11.0.


2. Run the following command to extract point features from the pretrained WaffleIron model (require 10883Mb GPU memory) pretrained on SemanticKITTI. The extracted features will be stored in the `result_folder`:
      ```
      cd main/WaffleIron_mod
      python extract_point_features.py \
      --path_dataset /gpfsdswork/dataset/SemanticKITTI \
      --ckpt pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
      --config configs/WaffleIron-48-256__kitti.yaml \
      --result_folder /lustre/fsn1/projects/rech/kvd/uyl37fq/main_preprocess/kitti/waffleiron_v2 \
      --phase val \
      --num_workers 3 \
      --num_votes 10 \
      --batch_size 2
      ```
### 4.2 Training
> [!NOTE]
> The generated instance label is supposed to be stored in `os.path.join(dataset_preprocess_root, "instance_labels_v2")`

cd main/
      python scripts/train.py
            --dataset_preprocess_root=/lustre/fsn1/projects/rech/kvd/uyl37fq/main_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI \
            --log_dir=logs \
            --exp_prefix=main_single --lr=1e-4 --seed=0 \
            --data_aug=True --max_angle=30.0 --translate_distance=0.2 \
            --enable_log=True \
            --n_infers=1
