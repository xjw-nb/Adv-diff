# the --num option value need to be set less than or equal to your src images numbers.
#python eval.py --model IR152 --dataset celeba --t 999 --save res --num 5
#python -m pytorch_fid res/img celeba-hq_sample/src
#python psnr_ssim.py --dir0 res/img --dir1 celeba-hq_sample/src

python eval.py --model resnet50 --dataset imagenet --t 999 --save res --num 1
python -m pytorch_fid res/img test/src
python psnr_ssim.py --dir0 res/img --dir1 test/src
