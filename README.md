This code is supplementary to the submitted paper with ID 50
Title: SimpleView++: Neighborhood Views for Point Cloud Classification
Authors: Shivanand Venkanna Sheshappanavar and Chandra Kambhamettu

#### Install Libraries
We recommend you first install [Anaconda](https://anaconda.org/) and create a virtual environment.

conda create --name svpp python=3.7.5 -y
conda activate svpp
pip install -r requirements.txt
conda install sed  -y
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.6"
cd pointnet2_pyt && pip install -e . && cd ..

#### Download Datasets and Pre-trained Models
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
unzip modelnet40_ply_hdf5_2048.zip
cd ..

## Running Experiments

#### Training
python main.py --exp-config configs/dgcnn_simpleview_run_1.yaml

#### Testing
python main.py --entry test --exp-config configs/dgcnn_simpleview_run_1.yaml --model-path runs2/dgcnn_simpleview_run_1/model_best_test.pth

Note: this is a basic configuration set at 32x32 resolutions: edit line 4 of mv_utils.py and line 50 of mv.py in models directory to increase the resolutions
We will provide detailed steps to recreate all results upon acceptance (we will also release the pre-trained models)

## Acknowlegements
We would like to thank the authors of the following repositories for sharing their code.
- SimpleView: Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline: [1](https://github.com/princeton-vl/SimpleView)
- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation: [1](https://github.com/charlesq34/pointnet), [2](https://github.com/fxia22/pointnet.pytorch)
- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space: [1](https://github.com/charlesq34/pointnet2), [2](https://github.com/erikwijmans/Pointnet2_PyTorch)
- Relation-Shape Convolutional Neural Network for Point Cloud Analysis: [1](https://github.com/Yochengliu/Relation-Shape-CNN)
- Dynamic Graph CNN for Learning on Point Clouds: [1](https://github.com/WangYueFt/dgcnn)
