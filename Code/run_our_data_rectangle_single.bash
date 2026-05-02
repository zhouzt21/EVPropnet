for dir in /hy-tmp/10000/*/; do
    name=$(basename "$dir")
    python Test1.py \
    --CheckPointPath=/home/EVPropNet/DVSProp_single/CheckPoints/79model.ckpt \
    --GPUDevice 0 \
    --WritePath=/home/EVPropNet/DVSProp_single/Outputs/our_data_10000_rectangle/$name/ \
    --TestMode=M \
    --Input="$dir" \
    --ImgFormat=png \
    --ImgSize [720,1280,1]
done