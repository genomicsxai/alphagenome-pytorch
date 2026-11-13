"""DeepLIFT attribution helpers."""

from torch import nn

from alphagenome_pytorch.layers import BilinearOp


def prepare_model_for_deeplift(model: nn.Module) -> nn.Module:
    """Replace local BilinearOp modules so tangermeme's bilinear rule applies.

    Modifies and returns the same model, preserving each contraction equation
    and module's training mode.
    """
    from tangermeme.deep_lift_shap import BilinearOp as TangermemeBilinearOp

    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, BilinearOp):
                replacement = TangermemeBilinearOp(child.equation)
                replacement.train(child.training)
                setattr(parent, name, replacement)

    return model
