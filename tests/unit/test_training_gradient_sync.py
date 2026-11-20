"""Regression coverage for aliased clipping and missing per-rank gradients."""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from alphagenome_pytorch.extensions.finetuning.gradient_utils import (
    average_gradients_across_ranks,
    unique_trainable_parameters,
)


class _TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, embeddings, organism_idx, **kwargs):
        return {1: torch.nn.functional.softplus(embeddings[1].transpose(1, 2)*self.weight)}

    def scale(self, targets, organism_idx, **kwargs):
        return targets


class _TinyFrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.head = _TinyHead()
        self.seen_organisms = []

    def forward(self, sequence, organism_idx, **kwargs):
        self.seen_organisms.append(organism_idx.detach().cpu().tolist())
        return {'embeddings_1bp': sequence.transpose(1, 2)*self.trunk}


def _training_loop_worker(rank, init_file, single_head=False):
    from alphagenome_pytorch.extensions.finetuning.training import train_epoch_ddp, train_epoch_multihead
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        model = _TinyFrozenModel()
        ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        def batch(worker, microbatch):
            sequence = torch.arange(1., 5.).reshape(1, 4, 1)+microbatch
            target = torch.full((1, 4, 1), float(10*worker+microbatch))
            return sequence, {'rna_seq': {1: target}}

        def train(current_model, head, batches, accumulation, world):
            optimizer = torch.optim.SGD(head.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            kwargs = dict(positional_weight=5., count_weight=1., epoch=1, log_every=100,
                use_amp=False, accumulation_steps=accumulation, frozen_backbone=True,
                num_segments=1, rank=rank, world_size=world, max_grad_norm=1.)
            if single_head:
                train_epoch_ddp(current_model, head,
                                [(seq, targets['rna_seq']) for seq, targets in batches],
                                optimizer, scheduler, torch.device('cpu'), {1: 1.}, **kwargs)
            else:
                train_epoch_multihead(
                    current_model, {'rna_seq': head}, batches, optimizer, scheduler,
                    torch.device('cpu'), {'rna_seq': 1.}, {'rna_seq': {1: 1.}}, **kwargs)

        # Real training loop: two local microbatches per rank, one optimizer step.
        train(ddp, model.head, [batch(rank, i) for i in range(2)], 2, 2)
        peer = model.head.weight.detach().clone()
        dist.broadcast(peer, src=0)
        torch.testing.assert_close(model.head.weight.detach(), peer, atol=0, rtol=0)
        # The data-parallel update must also match the single-process global
        # batch, not merely produce an identical but incorrectly scaled update.
        reference = _TinyFrozenModel()
        train(reference, reference.head,
              [batch(worker, i) for worker in range(2) for i in range(2)], 4, 1)
        torch.testing.assert_close(model.head.weight, reference.head.weight, atol=1e-6, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def _distributed_gradient_worker(rank, init_file):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        first, partial, unused = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
        strided = torch.nn.Parameter(torch.zeros(2, 2))
        first.grad = torch.full_like(first, rank+1)
        if rank == 0:
            partial.grad = torch.full_like(partial, 8)
        strided.grad = torch.full((2, 2), float(rank+2)).T
        assert not strided.grad.is_contiguous()
        parameters = [first, partial, unused, strided, first]
        average_gradients_across_ranks(parameters)
        torch.testing.assert_close(first.grad, torch.full_like(first, 1.5))
        torch.testing.assert_close(partial.grad, torch.full_like(partial, 4))
        torch.testing.assert_close(strided.grad, torch.full_like(strided, 2.5))
        assert unused.grad is None
        parameters = unique_trainable_parameters(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer = torch.optim.AdamW(parameters, lr=.1)
        optimizer.step()
        for parameter in parameters:
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter.detach(), reference, atol=0, rtol=0)
        assert unused not in optimizer.state
        # An all-unused update must leave gradients absent rather than create
        # zero gradients which would decay weights/update existing Adam moments.
        optimizer.zero_grad(set_to_none=True)
        average_gradients_across_ranks(parameters)
        assert all(parameter.grad is None for parameter in parameters)
    finally:
        dist.destroy_process_group()


class TestTrainingGradientSync(unittest.TestCase):
    def _check_multiple_batch_organisms(self, validation):
        from alphagenome_pytorch.extensions.finetuning.training import train_epoch_multihead, validate_multihead
        model = _TinyFrozenModel()
        batches = [
            (torch.ones(size, 4, 1), {'rna_seq': {1: torch.ones(size, 4, 1)}})
            for size in (2, 1)
        ]
        kwargs = dict(device=torch.device('cpu'), modality_weights={'rna_seq': 1.},
                      resolution_weights={'rna_seq': {1: 1.}}, positional_weight=5.,
                      count_weight=1., use_amp=False, num_segments=1, organism_idx=1)
        if validation:
            validate_multihead(model, {'rna_seq': model.head}, batches,
                               compute_pearson=False, **kwargs)
        else:
            optimizer = torch.optim.SGD(model.head.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            train_epoch_multihead(model, {'rna_seq': model.head}, batches, optimizer,
                                  scheduler, epoch=1, log_every=100, accumulation_steps=2,
                                  frozen_backbone=True, **kwargs)
        self.assertEqual(model.seen_organisms, [[1, 1], [1]])

    def test_training_keeps_scalar_organism_between_batches(self):
        self._check_multiple_batch_organisms(validation=False)

    def test_validation_keeps_scalar_organism_between_batches(self):
        self._check_multiple_batch_organisms(validation=True)

    def test_parameter_identity_and_frozen_filtering(self):
        head = torch.nn.Linear(2, 1)
        model = torch.nn.ModuleDict({'head': head})
        frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        source = [*head.parameters(), frozen, *model.parameters()]
        result = unique_trainable_parameters(p for p in source)
        self.assertEqual([id(p) for p in result], [id(p) for p in head.parameters()])

    def test_aliased_parameter_is_clipped_once(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3., 4.])
        torch.nn.utils.clip_grad_norm_(unique_trainable_parameters([parameter, parameter]), 1.)
        torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8]), rtol=1e-6, atol=1e-6)

    def test_single_process_is_noop(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        unused = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([2., 3.])
        average_gradients_across_ranks([parameter, unused])
        torch.testing.assert_close(parameter.grad, torch.tensor([2., 3.]), atol=0, rtol=0)
        self.assertIsNone(unused.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Requires Gloo')
    def test_distributed_partial_and_unused_gradients(self):
        with tempfile.TemporaryDirectory(prefix='alphagenome-grad-sync-') as directory:
            mp.spawn(_distributed_gradient_worker, args=(str(Path(directory) / 'rendezvous'),),
                     nprocs=2, join=True)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Requires Gloo')
    def test_frozen_training_loop_matches_global_batch(self):
        with tempfile.TemporaryDirectory(prefix='alphagenome-frozen-ddp-') as directory:
            mp.spawn(_training_loop_worker, args=(str(Path(directory) / 'rendezvous'),),
                     nprocs=2, join=True)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Requires Gloo')
    def test_single_head_frozen_loop_matches_global_batch(self):
        with tempfile.TemporaryDirectory(prefix='alphagenome-single-head-ddp-') as directory:
            mp.spawn(_training_loop_worker, args=(str(Path(directory) / 'rendezvous'), True),
                     nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
