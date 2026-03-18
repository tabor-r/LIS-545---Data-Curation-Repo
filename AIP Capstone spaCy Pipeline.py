#current working script to extract named entities from the oral history text and save them in a structured format with context examples. It processes each JSON file, extracts PERSON and ORG entities, counts their occurrences, and saves the results in a text file for each interview. Additionally, it creates a summary file that aggregates entity counts across all processed files.
#code was developed iteratively with the use of Claude AI.


import spacy
import pandas as pd
import json
import glob
import os
from collections import defaultdict

pd.options.display.max_rows = 600
pd.options.display.max_colwidth = 400
nlp = spacy.load('en_core_web_sm')

# Define file path and output folder
input_folder = r"/Users/rowantabor/Desktop/capstone/data/test_json"
output_folder = os.path.join(input_folder, "processed_texts")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Debugging print
print("Loading JSON files...")
files = glob.glob(os.path.join(input_folder, "*.json"))
if not files:
    print("No JSON files found! Check your directory path.")
else:
    print(f"Found {len(files)} files.")

# Track statistics across all files
# Structure: {(entity_text, label): {'count': int, 'contexts': [list of contexts]}}
all_entities = defaultdict(lambda: {'count': 0, 'contexts': []})
files_processed = 0

def get_context(doc, entity):
    """Extract 2 words before and 2 words after the entity"""
    # Find the token indices for the entity
    start_token = None
    end_token = None
    
    for i, token in enumerate(doc):
        if token.idx == entity.start_char:
            start_token = i
        if token.idx + len(token.text) == entity.end_char:
            end_token = i
            break
    
    if start_token is None or end_token is None:
        return f"... {entity.text} ..."
    
    # Get 2 words before and after
    context_start = max(0, start_token - 2)
    context_end = min(len(doc), end_token + 3)  # +3 because range is exclusive
    
    context_tokens = doc[context_start:context_end]
    context_text = " ".join([token.text for token in context_tokens])
    
    return context_text

# Process each JSON file
for filepath in files:
    filename = os.path.splitext(os.path.basename(filepath))[0]
    output_file_path = os.path.join(output_folder, f"{filename}_entities.txt")
    
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Get interview subject name
        interviewee_name = data.get('field_interviewee_name', 'Unknown')
        
        text = data.get('body', '')
        if not text:
            print(f"Warning: No 'body' found in {filepath}")
            continue
        
        document = nlp(text)
        
        # Collect entities from this file with context
        file_entities = defaultdict(lambda: {'count': 0, 'contexts': []})
        
        for ent in document.ents:
            # Filter to only include PERSON and ORG entities
            if ent.label_ in ['PERSON', 'ORG']:
                key = (ent.text, ent.label_)
                context = get_context(document, ent)
                file_entities[key]['count'] += 1
                file_entities[key]['contexts'].append(context)
                
                # Add to global tracker
                all_entities[key]['count'] += 1
                all_entities[key]['contexts'].append(context)
        
        # Sort by count
        sorted_entities = sorted(file_entities.items(), key=lambda x: x[1]['count'], reverse=True)
        
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            # Write interview subject as first line
            out_file.write(f"Interview Subject: {interviewee_name}\n")
            out_file.write(f"Source: {filename}\n")
            out_file.write(f"Total PERSON/ORG entities: {sum(data['count'] for data in file_entities.values())}\n")
            out_file.write("=" * 120 + "\n\n")
            
            # Write entities sorted by count (highest to lowest)
            out_file.write(f"{'ENTITY':<25} {'LABEL':<15} {'COUNT':<8} CONTEXT EXAMPLES\n")
            out_file.write("-" * 120 + "\n")
            
            for (ent_text, ent_label), data in sorted_entities:
                count = data['count']
                # Show up to 3 context examples
                contexts_to_show = data['contexts'][:3]
                
                # First line with entity info
                out_file.write(f"{ent_text:<25} {ent_label:<15} {count:<8} {contexts_to_show[0]}\n")
                
                # Additional context examples (indented)
                for context in contexts_to_show[1:]:
                    out_file.write(f"{'':<25} {'':<15} {'':<8} {context}\n")
                
                out_file.write("\n")  # Blank line between entities
        
        files_processed += 1
        print(f"✓ Saved {sum(data['count'] for data in file_entities.values())} PERSON/ORG entities to {output_file_path}")
        
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

# Create summary file with counts across all files
summary_file_path = os.path.join(output_folder, "_SUMMARY_all_entities.txt")
sorted_all_entities = sorted(all_entities.items(), key=lambda x: x[1]['count'], reverse=True)

with open(summary_file_path, "w", encoding="utf-8") as summary_file:
    summary_file.write("SUMMARY: ALL PERSON/ORG ENTITIES ACROSS ALL FILES\n")
    summary_file.write("=" * 120 + "\n")
    summary_file.write(f"Total files processed: {files_processed}\n")
    summary_file.write(f"Total entity mentions: {sum(data['count'] for data in all_entities.values())}\n")
    summary_file.write(f"Unique entities: {len(all_entities)}\n")
    summary_file.write("=" * 120 + "\n\n")
    
    summary_file.write(f"{'ENTITY':<25} {'LABEL':<15} {'COUNT':<8} CONTEXT EXAMPLES\n")
    summary_file.write("-" * 120 + "\n")
    
    for (ent_text, ent_label), data in sorted_all_entities:
        count = data['count']
        # Show up to 3 context examples from across all files
        contexts_to_show = data['contexts'][:3]
        
        # First line with entity info
        summary_file.write(f"{ent_text:<25} {ent_label:<15} {count:<8} {contexts_to_show[0]}\n")
        
        # Additional context examples (indented)
        for context in contexts_to_show[1:]:
            summary_file.write(f"{'':<25} {'':<15} {'':<8} {context}\n")
        
        summary_file.write("\n")  # Blank line between entities

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"Files processed: {files_processed}/{len(files)}")
print(f"Total PERSON/ORG entity mentions: {sum(data['count'] for data in all_entities.values())}")
print(f"Unique PERSON/ORG entities: {len(all_entities)}")
print(f"All outputs saved in {output_folder}")
print(f"Summary file: {summary_file_path}")