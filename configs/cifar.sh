python -m torch.distributed.launch --nproc_per_node=2 --master_port 29534 --use_env main.py --dist-eval --data-path cifar-100-python --model resnet18  --output_dir ./log/cifar/resnet18  --batch-size 100  --data-set CIFAR --lr 0.1 --epochs 800 --weight-decay 0.0001 --opt sgd