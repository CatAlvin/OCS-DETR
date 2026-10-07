# ===== 基本配置 =====
dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip       # 可包含 "slowfast" / "clip" / "blip" 的组合
t_feat_type=clip                 # 可选 clip|blip（与你另一份保持一致）
a_feat_type=pann
results_root=results
exp_id=exp

# ===== data paths =====
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

# ===== features root 指向 qvhighlight 子目录（与另一份一致）=====
feat_root=../features/qvhighlight

# ===== video features =====
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=(${feat_root}/blip_features)
  (( v_feat_dim += 512 ))
fi

# ===== text features（与另一份保持一致的判断分支）=====
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features
  t_feat_dim=512
elif [[ ${t_feat_type} == "blip" ]]; then
  # 复用 clip_text_features（和你另一份一致）
  t_feat_dir=${feat_root}/clip_text_features
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type. Use 'clip' or 'blip'."
  exit 1
fi

# ===== audio features（不改动逻辑，仅跟随 feat_root 路径）=====
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/umt_pann_features/
  a_feat_dim=2050
else
  echo "Wrong arg for a_feat_type."
  exit 1
fi

# ===== 位置嵌入可选项（与另一份一致）=====
# 可选：sine | learned | rope
pos_emb=${pos_emb:-sine}
# 文本侧：learned | rope | none（需配合 use_txt_pos=1 才生效）
txt_pos_emb=${txt_pos_emb:-learned}
# 是否给文本加位置编码：1/0
use_txt_pos=${use_txt_pos:-1}
# RoPE 相关
rope_base=${rope_base:-10000}
hidden_dim=${hidden_dim:-256}
max_v_l=${max_v_l:-75}

# 组装位置嵌入参数（显式字符串）
PE_ARGS="--position_embedding ${pos_emb} --rope_base ${rope_base} --hidden_dim ${hidden_dim} --max_v_l ${max_v_l}"
if [[ "${use_txt_pos}" == "1" ]]; then
  PE_ARGS="${PE_ARGS} --use_txt_pos --txt_position_embedding ${txt_pos_emb}"
fi

# ===== training =====
bsz=32

PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
  --dset_name ${dset_name} \
  --ctx_mode ${ctx_mode} \
  --train_path ${train_path} \
  --eval_path ${eval_path} \
  --eval_split_name ${eval_split_name} \
  --v_feat_dirs ${v_feat_dirs[@]} \
  --v_feat_dim ${v_feat_dim} \
  --t_feat_dir ${t_feat_dir} \
  --t_feat_dim ${t_feat_dim} \
  --a_feat_dir ${a_feat_dir} \
  --a_feat_dim ${a_feat_dim} \
  --bsz ${bsz} \
  --results_root ${results_root} \
  --exp_id ${exp_id} \
  ${PE_ARGS} \
  ${@:1}
