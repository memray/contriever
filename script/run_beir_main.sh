#!/usr/bin/env bash

MODEL_NAME=princeton-nlp/sup-simcse-roberta-large
MODEL_SNAME=sup-simcse-roberta-large
#MODEL_NAME=sentence-transformers/msmarco-roberta-base-ance-firstp
#MODEL_NAME=princeton-nlp/unsup-simcse-roberta-large
#MODEL_NAME=princeton-nlp/sup-simcse-roberta-large
#MODEL_NAME=princeton-nlp/unsup-simcse-roberta-base
#MODEL_NAME=princeton-nlp/unsup-simcse-roberta-base
#MODEL_NAME=princeton-nlp/unsup-simcse-bert-large-uncased
#MODEL_NAME=princeton-nlp/unsup-simcse-roberta-large
#MODEL_NAME=sentence-transformers/msmarco-bert-base-dot-v5
#MODEL_NAME=facebook/contriever-msmarco

OUTPUT_BASE_DIR=/export/home/exp/search/contriever
BATCH_SIZE=64
norm_query=False
norm_doc=False
score_function='cos_sim' # dot or cos_sim

EXP_NAME="$MODEL_SNAME.$score_function"
if [ $norm_query = True ] ; then
  EXP_NAME="$EXP_NAME.NormQ"
fi
if [ $norm_doc = True ] ; then
  EXP_NAME="$EXP_NAME.NormD"
fi

OUTPUT_DIR="$OUTPUT_BASE_DIR/$EXP_NAME"
mkdir -p $OUTPUT_DIR

DATA_DIR=/export/home/data/beir/

DATASETS=(msmarco-v2 bioasq msmarco fever climate-fever hotpotqa dbpedia-entity signal1m nq robust04 quora trec-news cqadupstack webis-touche2020 trec-covid fiqa arguana scidocs scifact nfcorpus) # all datasets, sorted by size
DATASETS=(trec-covid nfcorpus nq hotpotqa fiqa arguana webis-touche2020 quora cqadupstack dbpedia scidocs fever climate-fever scifact) # used in Contriever
DATASETS=(nq) # Table 1, QA datasets, TriviaQA is not included yet, report R@5/20/100
DATASETS=(trec-covid nfcorpus nq hotpotqa fiqa arguana webis-touche2020 quora cqadupstack dbpedia scidocs fever climate-fever scifact) # Table 2/8/9, BEIR datasets, report nDCG@10 R@100
DATASETS=(scifact nfcorpus fiqa) # Table 3, fewshot datasets, report nDCG@10
DATASETS=(nfcorpus nq fiqa arguana quora dbpedia scidocs fever) # Table 4/5/6 w/o fine-tuning, Table 7 w/FT, report nDCG@10

# 8 small datasets
#DATASETS=(trec-covid nfcorpus fiqa arguana webis-touche2020 cqadupstack scidocs scifact)
#CUDA_VISIBLE_DEVICES=(0 1 2 3 4 5 6 7)
#CUDA_VISIBLE_DEVICES=(8 9 10 11 12 13 14 15)

# 4 mid-size datasets
#DATASETS=(hotpotqa dbpedia-entity nq quora)
#CUDA_VISIBLE_DEVICES=(0 1 2 3 4 5 6 7)
#CUDA_VISIBLE_DEVICES=(8 9 10 11)

# 1 large-size dataset MSMARCO
DATASETS=(hotpotqa dbpedia-entity nq quora fever climate-fever msmarco)
CUDA_VISIBLE_DEVICES=(8 9 10 11 12 13 14 15)
#CUDA_VISIBLE_DEVICES=(2 3 4 5 6 7)


i=0
for DATASET in "${DATASETS[@]}"
do
#  echo ${CUDA_VISIBLE_DEVICES[${i}]}
  cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES[${i}]} nohup python eval_beir.py --model_name_or_path $MODEL_NAME --dataset $DATASET --beir_data_path $DATA_DIR --per_gpu_batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR --metric $score_function"
  if [ $norm_query = True ] ; then
    cmd="$cmd --norm_query"
  fi
  if [ $norm_doc = True ] ; then
    cmd="$cmd --norm_doc"
  fi
  NOHUP_PATH="$OUTPUT_BASE_DIR/$EXP_NAME/$DATASET.nohup"
  cmd="$cmd > $NOHUP_PATH &"

  i=$((i+1))
  if [ $i -eq "${#CUDA_VISIBLE_DEVICES[@]}" ]; then
      i=0
  fi
  echo $cmd
  eval $cmd
done
