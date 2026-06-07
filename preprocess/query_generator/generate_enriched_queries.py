import sys
import os
import pickle

# Add parent directory to path to import enriched_eval module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_generator import generate_enriched_queries
import argparse
import json
import pandas as pd

# Default API key (can be overridden via command line)
DEFAULT_API_KEY = "YOUR_OPENAI_API_KEY"

def load_msrvtt_csv(csv_path):
    print(f"Loading MSRVTT CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    data = {}
    for _, row in df.iterrows():
        key = row['key']
        data[key] = {
            'vid_key': row['vid_key'],
            'video_id': row['video_id'],
            'sentence': row['sentence']
        }
    
    print(f"Loaded {len(data)} MSRVTT queries")
    return data

def load_msvd_data(test_list_path, raw_captions_path):
    print(f"Loading MSVD test list from: {test_list_path}")
    with open(test_list_path, 'r') as f:
        test_videos = [line.strip() for line in f if line.strip()]
    
    print(f"Loading MSVD captions from: {raw_captions_path}")
    with open(raw_captions_path, 'rb') as f:
        all_captions = pickle.load(f)
    
    test_data = {}
    for video_id in test_videos:
        if video_id in all_captions:
            original_captions = all_captions[video_id]
            first_caption_text = ' '.join(original_captions[0])
            test_data[video_id] = {
                'original_captions': original_captions,
                'first_caption_text': first_caption_text
            }
    
    print(f"Loaded {len(test_data)} MSVD test videos")
    return test_data

def save_msrvtt_enriched_csv(enriched_data, original_data, output_csv_path, n_variations=10):
    print(f"Saving enriched MSRVTT CSV to: {output_csv_path}")
    rows = []
    for key in sorted(original_data.keys()):
        vid_key = original_data[key]['vid_key']
        video_id = original_data[key]['video_id']
        
        if key in enriched_data:
            captions = enriched_data[key]
        else:
            captions = [original_data[key]['sentence']] * (n_variations + 1)
        
        rows.append({
            'key': key,
            'vid_key': vid_key,
            'video_id': video_id,
            'sentence': captions[0]
        })
        
        for j in range(1, n_variations + 1):
            enriched_key = f"{key}_{j}"
            rows.append({
                'key': enriched_key,
                'vid_key': vid_key,
                'video_id': video_id,
                'sentence': captions[j] if j < len(captions) else captions[0]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(rows)} rows to {output_csv_path}")

def save_msvd_enriched_pkl(enriched_data, original_captions_dict, output_pkl_path):
    print(f"Saving enriched MSVD pickle to: {output_pkl_path}")
    output_data = {}
    metadata = {}
    
    for video_id, enriched_captions in enriched_data.items():
        all_captions_tokenized = []
        video_metadata = []
        
        if video_id in original_captions_dict:
            original_caps = original_captions_dict[video_id]
            num_originals = len(original_caps)
            
            for idx, original_cap_tokens in enumerate(original_caps):
                all_captions_tokenized.append(original_cap_tokens)
                group_start = len(all_captions_tokenized) - 1
                
                if idx == 0:
                    for enriched_text in enriched_captions[1:11]:
                        tokens = enriched_text.lower().split()
                        all_captions_tokenized.append(tokens)
                else:
                    for _ in range(10):
                        all_captions_tokenized.append(original_cap_tokens)
                
                group_end = len(all_captions_tokenized)
                video_metadata.append({
                    'original_index': idx,
                    'group_range': (group_start, group_end),
                    'original_text': ' '.join(original_cap_tokens),
                    'enriched_count': 10
                })
        else:
            print(f"Warning: {video_id} not in original_captions_dict")
            continue
        
        output_data[video_id] = all_captions_tokenized
        metadata[video_id] = {
            'num_original_captions': num_originals,
            'total_captions': len(all_captions_tokenized),
            'groups': video_metadata
        }
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    total_captions = sum(len(caps) for caps in output_data.values())
    print(f"Saved {len(output_data)} videos with {total_captions} total captions to {output_pkl_path}")
    return metadata

def save_reference_json(enriched_data, output_json_path):
    print(f"Saving reference JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    print(f"Saved reference data to {output_json_path}")

def save_standard_json(enriched_data, datatype, original_data, output_json_path):
    print(f"Saving standardized JSON to: {output_json_path}")
    unified_data = {}
    
    for key, captions in enriched_data.items():
        if not captions:
            continue
            
        original_cap = captions[0]
        aug_caps = captions[1:]
        
        if datatype == "msrvtt" and original_data is not None:
            video_id = original_data[key]['video_id']
            cap_id = key
        else:
            video_id = key
            cap_id = "cap_1"
            
        if video_id not in unified_data:
            unified_data[video_id] = {}
            
        unified_data[video_id][cap_id] = {
            "original": original_cap,
            "augment": aug_caps
        }
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(unified_data)} videos in standardized format to {output_json_path}")
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate enriched queries using GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--datatype", type=str, required=True,
                       choices=["msrvtt", "msvd", "didemo"],
                       help="Dataset type: msrvtt, msvd or didemo")
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input data file (CSV, test_list.txt, or JSON for didemo)")
    parser.add_argument("--raw_captions", type=str, default=None,
                       help="Path to raw-captions.pkl (MSVD only)")
    
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path (MSRVTT only)")
    parser.add_argument("--output_pkl", type=str, default=None,
                       help="Output pickle path (MSVD only)")
    parser.add_argument("--output_reference", type=str, default=None,
                       help="Output reference JSON path (for debugging)")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Output standardized JSON path (Applicable for all datasets)")

    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10, help="Number of variations per caption (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model to use (default: gpt-5-mini)")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Sleep time between API calls")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datatype == "msrvtt" and args.output_csv is None:
        parser.error("--output_csv is required for MSRVTT")
    elif args.datatype == "msvd":
        if args.raw_captions is None:
            parser.error("--raw_captions is required for MSVD")
        if args.output_pkl is None:
            parser.error("--output_pkl is required for MSVD")
    elif args.datatype == "didemo":
        if args.output_json is None and args.output_reference is None:
            print("Warning: Both --output_json and --output_reference are empty. You should specify at least one.")
    
    print("=== Enriched Query Generation ===")
    print(f"Dataset: {args.datatype.upper()}")
    print(f"Model: {args.model}")
    print(f"Variations per query: {args.n_variations}")
    
    input_queries = {}
    input_triplets = {}
    original_data = None
    original_queries_dict = {}

    if args.datatype == "msrvtt":
        print("\n[MSRVTT] Loading data...")
        original_data = load_msrvtt_csv(args.data_path)
        for key, data in original_data.items():
            input_queries[key] = data['sentence']
        print(f"Extracted {len(input_queries)} queries to enrich")
    elif args.datatype == "msvd":
        print("\n[MSVD] Loading data...")
        test_data = load_msvd_data(args.data_path, args.raw_captions)
        for video_id, video_data in test_data.items():
            input_queries[video_id] = video_data['first_caption_text']
            original_queries_dict[video_id] = video_data['original_captions']
        print(f"Extracted {len(input_queries)} videos")
    elif args.datatype == "didemo":
        print("\n[DiDeMo] Loading data...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            input_queries = json.load(f)
        print(f"Loaded {len(input_queries)} queries from DiDeMo JSON")
    
    if args.max_samples:
        print(f"\nLimiting to first {args.max_samples} samples for testing")
        input_queries = dict(list(input_queries.items())[:args.max_samples])
    
    print(f"\nTotal queries to enrich: {len(input_queries)}")
    
    if not args.output_reference:
        output_dir = os.path.dirname(args.output_json) if args.output_json else os.path.dirname(args.data_path)
        args.output_reference = os.path.join(output_dir, f"{args.datatype}_reference_data.json")

    print("\n=== Generating Enriched Queries ===")
    
    enriched_data = generate_enriched_queries(
        input_queries=input_queries,
        input_triplets=input_triplets,
        output_json_path=args.output_reference, 
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model,
        sleep_time=args.sleep_time
    )
    
    print("\nQuery generation completed!")
    print(f"Generated {len(enriched_data)} enriched video queries")
    
    print("\n=== Saving Output Files ===")
    if args.datatype == "msrvtt":
        save_msrvtt_enriched_csv(enriched_data, original_data, args.output_csv, args.n_variations)
        
        video_id_enriched_data = {}
        for key, queries in enriched_data.items():
            video_id = original_data[key]['video_id']
            if video_id not in video_id_enriched_data:
                video_id_enriched_data[video_id] = []
            video_id_enriched_data[video_id].extend(queries)
        save_reference_json(video_id_enriched_data, args.output_reference)
        
    elif args.datatype == "msvd":
        save_msvd_enriched_pkl(enriched_data, original_queries_dict, args.output_pkl)
        save_reference_json(enriched_data, args.output_reference)
        
    elif args.datatype == "didemo":
        if args.output_reference:
            save_reference_json(enriched_data, args.output_reference)

    if args.output_json:
        save_standard_json(enriched_data, args.datatype, original_data, args.output_json)

if __name__ == "__main__":
    main()