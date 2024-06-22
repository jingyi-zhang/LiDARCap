#!/bin/bash
python train.py --gpu 7 --datasets lidarcap_39 --eval_bs 4 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets lidarcap_7 --eval_bs 2 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets lidarcap_24 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets lidarcap_29 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets lidarcap_41 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_15 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_35 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_53 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_56 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_57 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets kitti_84 --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --datasets waymo --eval_bs 8 --threads 4 --epoch 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug


mkdir -p /home/ljl/exp/$2
mv /home/ljl/lidarcap/visual/$1/* /home/ljl/exp/$2/ 
mv /home/ljl/lidarcap/eval/$1/* /home/ljl/exp/$2/