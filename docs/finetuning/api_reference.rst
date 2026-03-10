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
