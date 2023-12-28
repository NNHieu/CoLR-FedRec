# FedMF base model
python fedtrain.py -m data=ml-1m net=fedmf net.init.gmf_emb_size=1,2,4,8,16,32,64 TRAIN.log_interval=10 EVAL.interval=100 TRAIN.wandb=True

# FedMF + SVD
python fedtrain.py -m data=ml-1m net=fedmf_svd net.init.gmf_emb_size=64 TRAIN.log_interval=10 EVAL.interval=100 compresor=svd net.compresor.rank=1,2,4,8,16,32 TRAIN.wandb=True

# FedMF + TopK
python fedtrain.py -m data=ml-1m net=fedmf_topk net.init.gmf_emb_size=64 TRAIN.log_interval=10 EVAL.interval=100 compresor=topk net.compresor.ratio=0.015625,0.03125,0.0625,0.125,0.25,0.5 TRAIN.wandb=True

# FedMF + CoLR
python fedtrain.py -m data=ml-1m net=fedmf_colr net.init.gmf_emb_size=64 net.init.rank=1,2,4,6,16,32 TRAIN.log_interval=10 EVAL.interval=100 TRAIN.wandb=True