eval "$(conda shell.bash hook)"
conda activate spino
mkdir -p logs
python semantic_fine_tuning.py fit --trainer.devices [6] --config configs/semantic_cityscapes.yaml > logs/semantic_cityscapes.txt 2>&1
python boundary_fine_tuning.py fit --trainer.devices [6] --config configs/boundary_cityscapes.yaml > logs/boundary_cityscapes.txt 2>&1
python instance_clustering.py test --trainer.devices [6,7] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes.txt 2>&1
