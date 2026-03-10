"""Weight mapping utilities for converting between PyTorch and JAX AlphaGenome models."""

from typing import Optional

def map_pytorch_to_jax(pt_name: str) -> Optional[str]:
    """Map a PyTorch parameter name to its corresponding JAX parameter name.
    
    Args:
        pt_name: Name of the parameter in PyTorch model.
        
    Returns:
        Corresponding JAX parameter name, or None if no mapping exists.
    """
    jax_key = None
    
    # --- Encoder ---
    if 'encoder.dna_embedder.conv1' in pt_name:
        # alphagenome/sequence_encoder/dna_embedder/conv1_d
        jax_prefix = 'alphagenome/sequence_encoder/dna_embedder/conv1_d'
        if 'weight' in pt_name: jax_key = f"{jax_prefix}/w"
        if 'bias' in pt_name: jax_key = f"{jax_prefix}/b"
        
    elif 'encoder.dna_embedder.block' in pt_name:
        # alphagenome/sequence_encoder/dna_embedder/conv_block
        jax_prefix = 'alphagenome/sequence_encoder/dna_embedder/conv_block'
        # sub = 'standardized_conv1_d' if 'conv' in pt_name else 'rms_batch_norm'
        # Logic from convert_weights / conftest
        if 'conv.weight' in pt_name: jax_key = f"{jax_prefix}/standardized_conv1_d/w"
        if 'conv.scale' in pt_name: jax_key = f"{jax_prefix}/standardized_conv1_d/scale"
        if 'conv.bias' in pt_name: jax_key = f"{jax_prefix}/standardized_conv1_d/bias"
        # Scales in PyTorch RMSNorm are 'weight', in JAX 'scale'
        if 'norm.weight' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/scale" 
        if 'norm.bias' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/offset"
        if 'norm.running_var' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/var_ema"
        
    elif 'encoder.down_blocks' in pt_name:
        # alphagenome/sequence_encoder/downres_block_X
        # pt: encoder.down_blocks.0.block1.conv.weight
        parts = pt_name.split('.')
        try:
            block_idx = int(parts[2])
            sub_block = parts[3] # block1 or block2
            
            jax_block_name = f"downres_block_{block_idx}"
            
            # block1 -> conv_block, block2 -> conv_block_1
            jax_sub = 'conv_block' if sub_block == 'block1' else 'conv_block_1'
            
            if 'conv' in parts:
                layer_type = 'standardized_conv1_d'
                if 'weight' in pt_name: suffix = 'w'
                elif 'scale' in pt_name: suffix = 'scale'
                elif 'bias' in pt_name: suffix = 'bias'
                else: suffix = 'unknown' # should not happen for known params
            else:
                layer_type = 'rms_batch_norm'
                if 'weight' in pt_name: suffix = 'scale'
                elif 'bias' in pt_name: suffix = 'offset'
                elif 'running_var' in pt_name: suffix = 'var_ema'
                else: suffix = 'unknown'
            
            jax_prefix = f"alphagenome/sequence_encoder/{jax_block_name}/{jax_sub}/{layer_type}"
            jax_key = f"{jax_prefix}/{suffix}"
        except (IndexError, ValueError):
            pass

    # --- Decoder ---
    elif 'decoder.up_blocks' in pt_name:
        parts = pt_name.split('.')
        try:
            block_idx = int(parts[2])
            jax_block_name = 'up_res_block' if block_idx == 0 else f"up_res_block_{block_idx}"
            
            pt_sub = parts[3] # conv_in, pointwise, conv_out
            
            if pt_sub == 'conv_in': jax_sub = 'conv_in'
            elif pt_sub == 'pointwise': jax_sub = 'pointwise_conv_unet_skip'
            elif pt_sub == 'conv_out': jax_sub = 'conv_out'
            else: jax_sub = 'unknown'
            
            if pt_sub == 'pointwise':
                # NCL refactor: proj -> conv
                if 'conv.weight' in pt_name: jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/linear/w"
                elif 'conv.bias' in pt_name: jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/linear/b"
                elif 'norm.weight' in pt_name: jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/rms_batch_norm/scale"
                elif 'norm.bias' in pt_name: jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/rms_batch_norm/offset"
                elif 'norm.running_var' in pt_name: jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/rms_batch_norm/var_ema"
            else:
                if 'conv' in parts:
                    layer_type = 'standardized_conv1_d'
                    if 'weight' in pt_name: suffix = 'w'
                    elif 'scale' in pt_name: suffix = 'scale'
                    elif 'bias' in pt_name: suffix = 'bias'
                    else: suffix = 'unknown'
                else:
                    layer_type = 'rms_batch_norm'
                    if 'weight' in pt_name: suffix = 'scale'
                    elif 'bias' in pt_name: suffix = 'offset'
                    elif 'running_var' in pt_name: suffix = 'var_ema'
                    else: suffix = 'unknown'
                
                jax_prefix = f"alphagenome/sequence_decoder/{jax_block_name}/{jax_sub}/{layer_type}"
                jax_key = f"{jax_prefix}/{suffix}"
            
            if 'residual_scale' in pt_name:
                jax_key = f"alphagenome/sequence_decoder/{jax_block_name}/residual_scale"
        except (IndexError, ValueError):
            pass

    # --- Tower ---
    elif 'tower.blocks' in pt_name:
        parts = pt_name.split('.')
        try:
            block_idx = int(parts[2])
            sub_mod = parts[3] # mha, mlp, attn_bias, pair_update
            
            jax_idx = '' if block_idx == 0 else f"_{block_idx}"
            
            if sub_mod == 'mha':
                jax_block = f"alphagenome/transformer_tower/mha_block{jax_idx}"
                if 'q_proj' in pt_name: jax_key = f"{jax_block}/q_layer/w"
                if 'k_proj' in pt_name: jax_key = f"{jax_block}/k_layer/w"
                if 'v_proj' in pt_name: jax_key = f"{jax_block}/v_layer/w"
                if 'out_proj.weight' in pt_name: jax_key = f"{jax_block}/linear_embedding/w"
                if 'out_proj.bias' in pt_name: jax_key = f"{jax_block}/linear_embedding/b"

                # Norms
                if 'norm.weight' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/scale"
                if 'norm.bias' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/offset"
                if 'norm.running_var' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/var_ema"

                # Q/K/V norms
                if 'norm_q.weight' in pt_name: jax_key = f"{jax_block}/norm_q/scale"
                if 'norm_q.bias' in pt_name: jax_key = f"{jax_block}/norm_q/offset"
                if 'norm_k.weight' in pt_name: jax_key = f"{jax_block}/norm_k/scale"
                if 'norm_k.bias' in pt_name: jax_key = f"{jax_block}/norm_k/offset"
                if 'norm_v.weight' in pt_name: jax_key = f"{jax_block}/norm_v/scale"
                if 'norm_v.bias' in pt_name: jax_key = f"{jax_block}/norm_v/offset"

                # Linear embedding (output projection)
                if 'linear_embedding.weight' in pt_name: jax_key = f"{jax_block}/linear_embedding/w"
                if 'linear_embedding.bias' in pt_name: jax_key = f"{jax_block}/linear_embedding/b"

                # Final norm (second RMS batch norm)
                if 'final_norm.weight' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/scale"
                if 'final_norm.bias' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/offset"
                if 'final_norm.running_var' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/var_ema"
                
            elif sub_mod == 'mlp':
                jax_block = f"alphagenome/transformer_tower/mlp_block{jax_idx}"
                if 'fc1.weight' in pt_name: jax_key = f"{jax_block}/linear/w"
                if 'fc1.bias' in pt_name: jax_key = f"{jax_block}/linear/b"
                if 'fc2.weight' in pt_name: jax_key = f"{jax_block}/linear_1/w"
                if 'fc2.bias' in pt_name: jax_key = f"{jax_block}/linear_1/b"
                if 'norm.weight' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/scale"
                if 'norm.bias' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/offset"
                if 'norm.running_var' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/var_ema"
                if 'final_norm.weight' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/scale"
                if 'final_norm.bias' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/offset"
                if 'final_norm.running_var' in pt_name: jax_key = f"{jax_block}/rms_batch_norm_1/var_ema"
                
            elif sub_mod == 'attn_bias':
                 jax_block = f"alphagenome/transformer_tower/attention_bias_block{jax_idx}"
                 if 'proj.weight' in pt_name: jax_key = f"{jax_block}/linear/w"
                 if 'norm.weight' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/scale"
                 if 'norm.bias' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/offset"
                 if 'norm.running_var' in pt_name: jax_key = f"{jax_block}/rms_batch_norm/var_ema"
                 
            elif sub_mod == 'pair_update':
                 # Only for even blocks.
                 pair_idx = block_idx // 2
                 jax_pair_idx = '' if pair_idx == 0 else f"_{pair_idx}"
                 jax_block = f"alphagenome/transformer_tower/pair_update_block{jax_pair_idx}"
                 
                 if 'seq2pair' in pt_name:
                     sub_prefix = f"{jax_block}/sequence_to_pair_block"
                     if 'norm_seq2pair.weight' in pt_name: jax_key = f"{sub_prefix}/norm_seq2pair/scale"
                     if 'norm_seq2pair.bias' in pt_name: jax_key = f"{sub_prefix}/norm_seq2pair/offset"
                     
                     if 'linear_q.weight' in pt_name: jax_key = f"{sub_prefix}/linear_q/w"
                     if 'linear_k.weight' in pt_name: jax_key = f"{sub_prefix}/linear_k/w"
                     if 'linear_pos_features.weight' in pt_name: jax_key = f"{sub_prefix}/linear_pos_features/w"
                     if 'linear_pos_features.bias' in pt_name: jax_key = f"{sub_prefix}/linear_pos_features/b"
                     
                     if 'q_r_bias' in pt_name: jax_key = f"{sub_prefix}/q_r_bias"
                     if 'k_r_bias' in pt_name: jax_key = f"{sub_prefix}/k_r_bias"
                     
                     if 'linear_y_q.weight' in pt_name: jax_key = f"{sub_prefix}/linear_y_q/w"
                     if 'linear_y_k.weight' in pt_name: jax_key = f"{sub_prefix}/linear_y_k/w"
                     if 'linear_pair.weight' in pt_name: jax_key = f"{sub_prefix}/linear_pair/w"
                     if 'linear_pair.bias' in pt_name: jax_key = f"{sub_prefix}/linear_pair/b"
                 
                 elif 'row_attn' in pt_name:
                     sub_prefix = f"{jax_block}/row_attention_block"
                     if 'norm.weight' in pt_name: jax_key = f"{sub_prefix}/layer_norm/scale"
                     if 'norm.bias' in pt_name: jax_key = f"{sub_prefix}/layer_norm/offset"
                     
                     if 'linear_q.weight' in pt_name: jax_key = f"{sub_prefix}/linear_q/w"
                     if 'linear_k.weight' in pt_name: jax_key = f"{sub_prefix}/linear_k/w"
                     if 'linear_v.weight' in pt_name: jax_key = f"{sub_prefix}/linear_v/w"
                     if 'linear_v.bias' in pt_name: jax_key = f"{sub_prefix}/linear_v/b"
                     
                 elif 'pair_mlp' in pt_name:
                     sub_prefix = f"{jax_block}/pair_mlp_block"
                     if 'norm.weight' in pt_name: jax_key = f"{sub_prefix}/layer_norm/scale"
                     if 'norm.bias' in pt_name: jax_key = f"{sub_prefix}/layer_norm/offset"
                     
                     if 'linear1.weight' in pt_name: jax_key = f"{sub_prefix}/linear/w"
                     if 'linear1.bias' in pt_name: jax_key = f"{sub_prefix}/linear/b"
                     if 'linear2.weight' in pt_name: jax_key = f"{sub_prefix}/linear_1/w"
                     if 'linear2.bias' in pt_name: jax_key = f"{sub_prefix}/linear_1/b"

        except (IndexError, ValueError):
            pass

    # --- Output Embedders ---
    elif 'embedder_128bp' in pt_name:
        # JAX: alphagenome/output_embedder
        jax_prefix = "alphagenome/output_embedder"
        if 'project_in.weight' in pt_name: jax_key = f"{jax_prefix}/linear/w"
        if 'project_in.bias' in pt_name: jax_key = f"{jax_prefix}/linear/b"
        if 'organism_embed.weight' in pt_name: jax_key = f"{jax_prefix}/embed/embeddings" 
        if 'norm.weight' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/scale"
        if 'norm.bias' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/offset"
        if 'norm.running_var' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/var_ema"

    elif 'embedder_1bp' in pt_name:
        # JAX: alphagenome/output_embedder_1
        jax_prefix = "alphagenome/output_embedder_1"
        if 'project_in.weight' in pt_name: jax_key = f"{jax_prefix}/linear/w"
        if 'project_in.bias' in pt_name: jax_key = f"{jax_prefix}/linear/b"
        if 'project_skip.weight' in pt_name: jax_key = f"{jax_prefix}/linear_1/w" # Skip projection
        if 'organism_embed.weight' in pt_name: jax_key = f"{jax_prefix}/embed/embeddings"
        if 'norm.weight' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/scale"
        if 'norm.bias' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/offset"
        if 'norm.running_var' in pt_name: jax_key = f"{jax_prefix}/rms_batch_norm/var_ema"

    elif 'embedder_pair' in pt_name:
        # JAX: alphagenome/output_pair
        jax_prefix = "alphagenome/output_pair"
        if 'organism_embed.weight' in pt_name: jax_key = f"{jax_prefix}/embed/embeddings"
        if 'norm.weight' in pt_name: jax_key = f"{jax_prefix}/layer_norm/scale"
        if 'norm.bias' in pt_name: jax_key = f"{jax_prefix}/layer_norm/offset"

    # --- Organism Embed ---
    elif 'organism_embed' in pt_name:
        # alphagenome/embed
         if 'weight' in pt_name: jax_key = "alphagenome/embed/embeddings"

    # --- Heads ---
    elif 'heads.' in pt_name:
        parts = pt_name.split('.')
        try:
            head_name = parts[1]
            
            if 'convs' in pt_name:
                # heads.atac.convs.1.weight (NCL format)
                res = parts[3]
                jax_prefix = f"alphagenome/head/{head_name}/resolution_{res}/multi_organism_linear"
                if 'weight' in pt_name: jax_key = f"{jax_prefix}/w"
                if 'bias' in pt_name: jax_key = f"{jax_prefix}/b"
                 
            elif 'residual_scales' in pt_name:
                 # heads.atac.residual_scales.1
                 res = parts[3]
                 jax_key = f"alphagenome/head/{head_name}/resolution_{res}/learnt_scale"
        except (IndexError, ValueError):
             pass

    # --- Contact Maps Head ---
    elif 'contact_maps_head' in pt_name:
        jax_prefix = "alphagenome/head/contact_maps/multi_organism_linear"
        if 'weight' in pt_name: jax_key = f"{jax_prefix}/w"
        if 'bias' in pt_name: jax_key = f"{jax_prefix}/b"

    # --- Splice Sites Classification Head ---
    elif 'splice_sites_classification_head' in pt_name:
        jax_prefix = "alphagenome/head/splice_sites_classification/multi_organism_linear"
        if 'conv.weight' in pt_name: jax_key = f"{jax_prefix}/w"
        elif 'conv.bias' in pt_name: jax_key = f"{jax_prefix}/b"

    # --- Splice Sites Usage Head ---
    elif 'splice_sites_usage_head' in pt_name:
        jax_prefix = "alphagenome/head/splice_sites_usage/multi_organism_linear"
        if 'conv.weight' in pt_name: jax_key = f"{jax_prefix}/w"
        elif 'conv.bias' in pt_name: jax_key = f"{jax_prefix}/b"

    # --- Splice Sites Junction Head ---
    elif 'splice_sites_junction_head' in pt_name:
        if 'conv.weight' in pt_name:
            jax_key = "alphagenome/head/splice_sites_junction/multi_organism_linear/w"
        elif 'conv.bias' in pt_name:
            jax_key = "alphagenome/head/splice_sites_junction/multi_organism_linear/b"
        elif 'rope_params' in pt_name:
            if 'pos_acceptor' in pt_name:
                jax_key = "alphagenome/head/splice_sites_junction/pos_acceptor_logits/embeddings"
            elif 'pos_donor' in pt_name:
                 jax_key = "alphagenome/head/splice_sites_junction/pos_donor_logits/embeddings"
            elif 'neg_acceptor' in pt_name:
                 jax_key = "alphagenome/head/splice_sites_junction/neg_acceptor_logits/embeddings"
            elif 'neg_donor' in pt_name:
                 jax_key = "alphagenome/head/splice_sites_junction/neg_donor_logits/embeddings"
    
    return jax_key
