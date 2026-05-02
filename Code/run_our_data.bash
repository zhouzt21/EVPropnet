for dir in /hy-tmp/700m-1500m/*/; do
    name=$(basename "$dir")
    python Test.py \
    --CheckPointPath=/home/EVPropNet/DVSProp/CheckPoints/49model.ckpt \
    --GPUDevice 0 \
    --WritePath=/home/EVPropNet/DVSProp/Outputs/our_data_700m-1500m/$name/ \
    --TestMode=M \
    --Input="$dir" \
    --ImgFormat=png \
    --ImgSize [720,1280,1]
done