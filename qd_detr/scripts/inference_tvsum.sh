ckpt_path=$1
eval_split_name=$2
eval_path=data/tvsum/tvsum_${eval_split_name}.jsonl

echo "Checkpoint: ${ckpt_path}"
echo "Eval Split: ${eval_split_name}"
echo "Eval Path:  ${eval_path}"

PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
  --resume ${ckpt_path} \
  --eval_split_name ${eval_split_name} \
  --eval_path ${eval_path} \
  ${@:3}
