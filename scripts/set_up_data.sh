## 1. Load basic competition data
echo "Loading basic competition data..."
uv run python kiva-iccv/utils/download.py

## 2. Create subimages for the training, validation, and test sets
echo "Creating subimages for the training, validation, and test sets..."
for dataset in train validation test unit; do
    uv run python kiva-iccv/utils/transform.py --dataset $dataset
done

## 3. Download the untransformed images for the on-the-fly dataset
echo "Downloading the untransformed images for the on-the-fly dataset..."
mkdir -p KiVA
mkdir -p data/KiVA
cd KiVA
git clone --depth 1 --branch main https://github.com/ey242/KiVA.git
cp -r "KiVA/untransformed objects/" ../data/KiVA/
cd ../
rm -rf KiVA


## 4. End!
echo "✅ Done! ✅"