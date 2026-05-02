for dir in /hy-tmp/100000/*/; do
    name=$(basename "$dir")
    python Test1.py \
    --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt \
    --GPUDevice 0 \
    --WritePath=/home/EVPropNet/DVSProp/Outputs/our_data_100000_rectangle/$name/ \
    --TestMode=M \
    --Input="$dir" \
    --ImgFormat=png \
    --ImgSize [720,1280,1]
done