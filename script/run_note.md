conda deactivate
conda activate ir

# BEIR Datasets
### 1 in-domain dataset, msmarco (inference is problematic, CUDA error) 
### 14 OOD available datasets, 4 to-go (bioasq, signal1m, robust04, trec-news)
arguana
climate-fever
cqadupstack
dbpedia-entity
fever
fiqa
hotpotqa
nfcorpus
nq
quora
scidocs
scifact
trec-covid
webis-touche2020



### single gpu
cd /export/share/ruimeng/project/search/contriever
source script/run_beir_main.sh

### distributed multi-gpu, 
#### NQ: bs=512, memory usage=20-30gb, takes 20min
cd /export/share/ruimeng/project/search/contriever
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;RANK=0;WORLD_SIZE=8;MASTER_PORT=7777;MASTER_ADDR=localhost nohup python -m torch.distributed.launch --nproc_per_node=8 eval_beir.py --model_name_or_path facebook/contriever-msmarco --dataset msmarco --beir_data_path /export/home/data/beir/ --per_gpu_batch_size 32 --output_dir /export/home/exp/search/contriever/fb-contriever.msmarco.dot > /export/home/exp/search/contriever/fb-contriever.msmarco.dot/nohup.output 2>&1 &


#### msmarco: 8gpu always cuda error, bs=512/256/32. 4gpu works (4,5,6,7)
[03/21/2022 02:10:15] {exact_search.py:53} INFO - Encoding Batch 89/177...
terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: unspecified launch failure