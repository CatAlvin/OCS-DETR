# 传入参数：1 = 路径模板，2 = split（val/test）
# 示例模板：
# "results_{domain}/tvsum-video_tef-exp-2025_06_11_17_48_47/model_best.ckpt"

template_path=$1
eval_split_name=$2

# 所有 TVSum domain
domains=(VT VU GA MS PK PR FM BK BT DS)

for domain in "${domains[@]}"; do
  echo "======= Running for domain: $domain ======="

  # 替换路径模板中的 {domain}
  ckpt_path=${template_path//\{domain\}/$domain}

  # 调用你已有的 inference_tvsum.sh
  bash qd_detr/scripts/inference_tvsum.sh "$ckpt_path" "$eval_split_name"
done
