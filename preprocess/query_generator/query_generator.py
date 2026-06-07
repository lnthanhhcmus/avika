import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

PROMPT_TEMPLATE = """You are given a user video-search query and a set of related knowledge graph triplets (head-tail-relation).
Your task is to generate EXACTLY {n} rewritten versions of the query, following the constraints below:

1. Each rewritten sentence must preserve the exact core semantic meaning of the original query.
2. Use provided Knowledge Graph triplets as limited supplementary context to ground query entities when relevant, without overriding the query intent.
3. Do NOT introduce new objects, actions, attributes, intentions, or context, and use only the given triplets to support what is explicitly stated in the original query.
4. Sentence structure and lexical choices should be diversified across rewritten sentences.
5. The rewritten queries should vary in length and number of words.
6. Each rewritten query must be no more than 10 words longer than the original query.
7. The original query must NOT appear in the rewritten outputs.
8. No two rewritten queries may be identical.
9. You may use periods (.) and commas (,) to clearly separate clauses or multiple actions, but do NOT use quotation marks ("), semicolons (;), or colons (:).

The input query is: "{query}"
The KG context (Triplets): "{triplets}"

Output ONLY the {n} rewritten sentences, one per line, without numbering, bullet points, or any additional explanations.
"""

def generate_enriched_queries(
    input_queries,
    input_triplets,
    output_json_path,
    api_key,
    n_variations=10,
    model="gpt-4.1",
    batch_size=1,
    sleep_time=1.0
):
    client = OpenAI(api_key=api_key)
    
    if isinstance(input_queries, list):
        input_queries = {f"video_{i}": q for i, q in enumerate(input_queries)}
    
    enriched_data = {}
    
    if os.path.exists(output_json_path):
        print(f"Loading existing enriched data from {output_json_path}")
        with open(output_json_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
    
    total_queries = len(input_queries)
    already_processed = len(enriched_data)
    remaining = total_queries - already_processed
    
    print(f"Generating enriched queries for {total_queries} queries...")
    print(f"Using model: {model}, variations per query: {n_variations}")
    if already_processed > 0:
        print(f"Already processed: {already_processed}, Remaining: {remaining}")
    
    processed_count = 0
    for video_id, original_query in tqdm(input_queries.items(), desc="Enriching queries"):
        if video_id in enriched_data:
            continue
        
        processed_count += 1
        
        try:
            triplets = input_triplets.get(video_id, "")
            prompt = PROMPT_TEMPLATE.format(n=n_variations, query=original_query, triplets=triplets)
            
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=750
            )
            
            variations_text = response.output_text.strip()
            raw_variations = [
                line.strip() 
                for line in variations_text.split('\n') 
                if line.strip()  
            ]
            
            seen = set()
            variations = []
            for var in raw_variations:
                normalized = ' '.join(var.lower().split())
                original_normalized = ' '.join(original_query.lower().split())
                
                if normalized not in seen and normalized != original_normalized:
                    variations.append(var)
                    seen.add(normalized)
            
            if len(variations) < n_variations:
                shortage = n_variations - len(variations)
                print(f"Warning: Only got {len(variations)} unique variations for {video_id}, padding {shortage} with modified original")
                for i in range(shortage):
                    variations.append(original_query)
            elif len(variations) > n_variations:
                variations = variations[:n_variations]
            
            enriched_data[video_id] = [original_query] + variations
            
            if processed_count % 10 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2, ensure_ascii=False)
                print(f"\nProgress: {len(enriched_data)}/{total_captions} completed, saved checkpoint")
            
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            enriched_data[video_id] = [original_query] * (n_variations + 1)
            continue
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    total_captions = sum(len(caps) for caps in enriched_data.values())
    avg_per_video = total_captions / len(enriched_data) if enriched_data else 0
    
    print(f"\nEnrichment completed!")
    print(f"Enriched data saved to {output_json_path}")
    print(f"Total videos: {len(enriched_data)}")
    print(f"Total captions (including originals): {total_captions}")
    print(f"Average captions per video: {avg_per_video:.1f}")
    print(f"Expected: {n_variations + 1} (1 original + {n_variations} enriched)")
    
    return enriched_data


def load_enriched_queries(json_path):
    """
    Load pre-generated enriched queries from JSON file.
    
    Args:
        json_path: Path to enriched queries JSON
        
    Returns:
        dict: {video_id: [original_caption, variation1, ..., variation_n]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded enriched queries for {len(data)} videos from {json_path}")
    return data


if __name__ == "__main__":
    # Example usage for offline preprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enriched queries using GPT-4")
    parser.add_argument("--input_json", type=str, required=True, help="Input captions JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output enriched queries JSON file")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10, help="Number of variations per caption")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Load input captions
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_queries = json.load(f)
    
    # Generate enriched queries
    generate_enriched_queries(
        input_queries=input_queries,
        output_json_path=args.output_json,
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model
    )
