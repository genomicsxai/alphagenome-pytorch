import sys
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from alphagenome_research.model import dna_model
from alphagenome.models import dna_output
from alphagenome.data import genome
from alphagenome_research.io import genome as genome_io

# Add src to path if needed (though usually handled by env)
# sys.path.append(...)

def main():
    parser = argparse.ArgumentParser(description="Manual Sequence Extraction and Inference Demo")
    parser.add_argument("jax_checkpoint", help="Path to JAX checkpoint directory")
    args = parser.parse_args()

    print(f"Loading model from {args.jax_checkpoint}...")
    # Load model (this sets up the FastaExtractor based on default settings)
    model = dna_model.create(args.jax_checkpoint)
    
    # ---------------------------------------------------------
    # 1. Define Interval and Variant
    # ---------------------------------------------------------
    # Example from user request
    interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
    variant = genome.Variant(
        chromosome='chr22',
        position=36201698,
        reference_bases='A',
        alternate_bases='C',
    )
    organism = dna_model.Organism.HOMO_SAPIENS
    
    # ---------------------------------------------------------
    # 2. Extract Sequences (String)
    # ---------------------------------------------------------
    print("\n--- extracting sequences ---")
    extractor = model._get_fasta_extractor(organism)
    
    # Extract Reference Sequence for the whole interval
    ref_seq_str = extractor.extract(interval)
    print(f"Reference Sequence Length: {len(ref_seq_str)}")
    print(f"Sample (start): {ref_seq_str[:50]}...")
    
    # Extract Variant Sequences (Ref and Alt)
    # Helper from genome_io takes care of inserting the variant
    ref_seq_str_v, alt_seq_str_v = genome_io.extract_variant_sequences(
        interval, variant, extractor
    )
    # Note: ref_seq_str_v should match ref_seq_str usually, unless variant handling does something special padding
    print(f"Ref Sequence (Variant helper): {len(ref_seq_str_v)} bp")
    print(f"Alt Sequence (Variant helper): {len(alt_seq_str_v)} bp")
    
    # ---------------------------------------------------------
    # 3. Preprocess (One-Hot Encoding)
    # ---------------------------------------------------------
    print("\n--- one-hot encoding ---")
    encoder = model._one_hot_encoder
    
    # Encode Reference
    ref_one_hot = encoder.encode(ref_seq_str_v) # (S, 4)
    ref_batch = ref_one_hot[np.newaxis]         # (1, S, 4)
    
    # Encode Alternate
    alt_one_hot = encoder.encode(alt_seq_str_v)
    alt_batch = alt_one_hot[np.newaxis]
    
    print(f"Batch Shape: {ref_batch.shape}")
    
    # ---------------------------------------------------------
    # 4. Manual Forward Pass (JAX Internals)
    # ---------------------------------------------------------
    print("\n--- manual forward pass ---")
    
    # Prepare auxiliary inputs required by _predict
    organism_index = jnp.array([0], dtype=jnp.int32) # 0 for Human
    track_metadata = model._metadata[organism]
    strand_reindexing = jax.device_put(track_metadata.strand_reindexing)
    negative_strand_mask = jnp.array([interval.negative_strand], dtype=bool) # False here
    
    # Move inputs to device (optional explicit put, JAX handles usually)
    ref_device = jax.device_put(ref_batch)
    alt_device = jax.device_put(alt_batch)
    
    # Run Prediction on Reference
    print("Running Ref prediction...")
    ref_preds = model._predict(
        model._params,
        model._state,
        ref_device,
        organism_index,
        negative_strand_mask=negative_strand_mask,
        strand_reindexing=strand_reindexing
    )
    
    # Run Prediction on Alternate
    print("Running Alt prediction...")
    alt_preds = model._predict(
        model._params,
        model._state,
        alt_device,
        organism_index,
        negative_strand_mask=negative_strand_mask,
        strand_reindexing=strand_reindexing
    )
    
    print("\nSuccess! Outputs obtained.")
    if 'atac' in ref_preds: # Key might be upper case enum
        print("ATAC keys:", ref_preds[dna_output.OutputType.ATAC].keys())
        
    # ---------------------------------------------------------
    # 5. Export for PyTorch (Optional)
    # ---------------------------------------------------------
    # You can save the sequences to use in PyTorch
    # np.save('ref_seq.npy', ref_batch)
    # np.save('alt_seq.npy', alt_batch)
    # print("Saved sequences to .npy files for PyTorch usage.")

if __name__ == "__main__":
    main()
