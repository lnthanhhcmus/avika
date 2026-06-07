# AVIKA: Knowledge-Guided Video-Caption Alignment for Bridging Information Asymmetry in Text-Video Retrieval

This repository contains the code for AVIKA, a framework designed to bridge the inherent information asymmetry between sparse text queries and dense visual content in text-video retrieval. By leveraging knowledge-guided alignment, AVIKA significantly enhances semantic understanding. The proposed architecture integrates video-caption interaction, query-aware weighting and filtering, multi-granularity matching, and a hard-negative loss strategy, demonstrating strong performance across standard benchmarks including MSRVTT, MSVD, and DiDeMo.

**Note:** This work is currently under review.

## Requirements
This repository requires two distinct environments:
- `AVIKA`: For running the main framework.
```sh
conda install --yes -c pytorch pytorch=1.13.1 torchvision cudatoolkit=11.6
pip install opencv-python==4.9.0.80 numpy==1.23.0 ftfy regex tqdm boto3 requests pandas
pip install ruptures ujson coloredlogs ffmpeg-python decord thop
pip install git+https://github.com/openai/CLIP.git
```
- `LLaVa`: For preprocessing and frame-level caption generation. Please refer to the official GitHub repository [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main).

## Data Preparation

### For MSRVTT

The official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). 

For the convenience, you can also download the splits and captions by,
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in [sharing](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt) from *Frozen️ in Time*, i.e.,
```sh
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### For MSVD

Raw videos can be download from [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/). 

The splits and `raw_captions` can be found in the wonderful job [collaborative-experts](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/msvd/README.md). For the convenience, you can also download them by,
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
```


### For DiDeMo

Raw videos can be download from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments). The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo/README.md).

## Data Preprocessing

### Compress Video for Speed-up (optional)
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

### Generate frame-level captions

Before generating captions for each frame, you need to perform preprocessing on the raw video to extract the frames.

```sh
python preprocess/video_frame_extractor.py --raw_video_path [your_raw_video_folder_path] --extracted_frame_path [your_output_frame_path]
```

Based on the extracted video frames, use LLaVa to generate captions for each frame.

```sh
python preprocess/frame_captions_generator/frame_captions_generator.py --video_frames_path [your_frame_path] --video_id_list_path [your_video_id.json]
```

### Generate and select query

Run the following script to create multiple query variations for your dataset:

```sh
python preprocess/query_generator/generate_enriched_queries.py --datatype [dataset_name] --data_path [path_to_data] --output_json [path_to_save_generated.json] --api_key [your_api_key] --n_variations [num_generate_query] --model [selected_model]
```

Then, filter the generated output to select the top-k most relevant queries:

```sh
python preprocess/query_generator/fqs_selector_json.py --input_path [path_to_save_generated.json] --output_path [path_to_save_filtered.json] --k [number_selected_queries]
```


## How to Run

(1) Extract the video dataset into frames, and organize both the frame data and related metadata within the `datasets` folder

(2) Download the pretrained CLIP weight:
- CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
- CLIP (ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

(3) Running the scripts:

### MSRVTT - ViT/B-32

```sh
DATA_PATH=[Your MSRVTT data and videos path]
OUTPUT_PATH=[Directory to save model weights and checkpoints during training]

python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --do_train --num_thread_reader=4 --epochs=5 --batch_size=64 --n_display=10 \
  --train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
  --val_csv ${DATA_PATH}/msrvtt_data/MSRVTT_JSFUSION_test.csv  \
  --data_path ${DATA_PATH}/msrvtt_data/MSRVTT_data.json \
  --frame_caption_path ${DATA_PATH}/msrvtt_data/MSRVTT_frame_captions.json \
  --features_path ${DATA_PATH}/frames \
  --output_dir ${OUTPUT_PATH} \
  --lr 1e-4 --max_words 64 --max_frames 12 --batch_size_val 64 \
  --datatype msrvtt --expand_msrvtt_sentences \
  --feature_framerate 1 --coef_lr 1e-3 \
  --freeze_layer_num 0  --slice_framepos 2 \
  --loose_type --linear_patch 2d --sim_header seqTransf \
  --hard_negative_weighting 1.0 --nucleus_P 0.4 \
  --pretrained_clip_name ViT-B/32 \
  --video_data_type frames \
  --co_attention_block
```

### MSRVTT - ViT/B-16
```sh
DATA_PATH=[Your MSRVTT data and videos path]
OUTPUT_PATH=[Directory to save model weights and checkpoints during training]

python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --do_train --num_thread_reader=4 --epochs=5 --batch_size=64 --n_display=10 \
  --train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
  --val_csv ${DATA_PATH}/msrvtt_data/MSRVTT_JSFUSION_test.csv  \
  --data_path ${DATA_PATH}/msrvtt_data/MSRVTT_data.json \
  --frame_caption_path ${DATA_PATH}/msrvtt_data/MSRVTT_frame_captions.json \
  --features_path ${DATA_PATH}/frames \
  --output_dir ${OUTPUT_PATH} \
  --lr 1e-4 --max_words 64 --max_frames 12 --batch_size_val 64 \
  --datatype msrvtt --expand_msrvtt_sentences \
  --feature_framerate 1 --coef_lr 1e-3 \
  --freeze_layer_num 0  --slice_framepos 2 \
  --loose_type --linear_patch 2d --sim_header seqTransf \
  --hard_negative_weighting 1.5 --nucleus_P 0.4 \
  --pretrained_clip_name ViT-B/16 \
  --video_data_type frames \
  --co_attention_block
```

### MSVD
```sh
DATA_PATH=[Your MSVD data and videos path]
OUTPUT_PATH=[Directory to save model weights and checkpoints during training]

python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --do_train --num_thread_reader=4 --epochs=5 --batch_size=64 --n_display=10 \
  --data_path ${DATA_PATH}/msvd_data \
  --frame_caption_path ${DATA_PATH}/msvd_data/MSVD_frame_captions.json \
  --features_path ${DATA_PATH}/frames \
  --output_dir ${OUTPUT_PATH} \
  --lr 1e-4 --max_words 64 --max_frames 12 --batch_size_val 64 \
  --datatype msvd \
  --feature_framerate 1 --coef_lr 1e-3 \
  --freeze_layer_num 0  --slice_framepos 2 \
  --loose_type --linear_patch 2d --sim_header seqTransf \
  --hard_negative_weighting 1.0 --nucleus_P 0.4 \
  --pretrained_clip_name ViT-B/32 \
  --video_data_type frames \
  --co_attention_block
```

### DiDeMo
```sh
DATA_PATH=[Your DiDeMo data and videos path]
OUTPUT_PATH=[Directory to save model weights and checkpoints during training]

python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --do_train --num_thread_reader=4 --epochs=10 --batch_size=32 --n_display=10 \
  --data_path ${DATA_PATH}/didemo_data \
  --frame_caption_path ${DATA_PATH}/didemo_data/didemo_frame_captions.json \
  --features_path ${DATA_PATH}/frames \
  --output_dir ${OUTPUT_PATH} \
  --lr 1e-4 --max_words 64 --max_frames 32 --batch_size_val 32 \
  --datatype didemo \
  --feature_framerate 1 --coef_lr 1e-3 \
  --freeze_layer_num 0  --slice_framepos 2 \
  --loose_type --linear_patch 2d --sim_header seqTransf \
  --hard_negative_weighting 1.0 --nucleus_P 0.4 \
  --pretrained_clip_name ViT-B/32 \
  --video_data_type frames \
  --co_attention_block
```

**Note:** To enable augmented query data, you can add `--aug_json_path`, `--fqs_k`, and `--aggregation_strategy` to your execution script. These arguments are supported in both `--do_train` and `--do_eval` modes.

# Resources
The implementation of NarVid relies on resources from [CLIP](https://github.com/openai/CLIP "CLIP"), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip") and [Cap4Video](https://github.com/whwu95/Cap4Video "Cap4Video").
