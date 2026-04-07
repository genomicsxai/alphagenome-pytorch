API Reference
=============

Transfer Learning
-----------------

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.load_trunk

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.prepare_for_transfer

.. autoclass:: alphagenome_pytorch.extensions.finetuning.transfer.TransferConfig
   :members:
   :undoc-members:

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.add_head

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.remove_all_heads

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.count_trainable_params

Adapters
--------

.. autofunction:: alphagenome_pytorch.extensions.finetuning.adapters.get_adapter_params

.. autofunction:: alphagenome_pytorch.extensions.finetuning.adapters.merge_adapters

.. autoclass:: alphagenome_pytorch.extensions.finetuning.adapters.LoRA
   :members:
   :undoc-members:

.. autoclass:: alphagenome_pytorch.extensions.finetuning.adapters.Locon
   :members:
   :undoc-members:

.. autoclass:: alphagenome_pytorch.extensions.finetuning.adapters.IA3
   :members:
   :undoc-members:

.. autoclass:: alphagenome_pytorch.extensions.finetuning.adapters.AdapterHoulsby
   :members:
   :undoc-members:

Checkpointing
-------------

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.save_checkpoint

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.load_checkpoint

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.save_delta_checkpoint

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.load_delta_checkpoint

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.export_delta_weights

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.load_delta_config

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.load_delta_weights

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.get_adapter_state_dict

.. autofunction:: alphagenome_pytorch.extensions.finetuning.checkpointing.compute_base_model_hash

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.transfer_config_to_dict

.. autofunction:: alphagenome_pytorch.extensions.finetuning.transfer.transfer_config_from_dict
