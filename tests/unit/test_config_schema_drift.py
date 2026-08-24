"""Guard: every YAML key the docs advertise must actually reach the trainer.

``docs/finetuning/cli.rst`` publishes a full config schema. Nothing previously
tied that schema to ``postprocess_args``, and the two drifted: ``save_delta``
and ``no_full_checkpoint`` were documented for months while the merge silently
ignored them, so a run the user believed was writing shareable delta
checkpoints wrote none.

This test fails if a documented key is not merged, or if a key listed below as
deliberately non-scalar has quietly become a plain scalar merge.
"""

import re
from pathlib import Path

import pytest
import yaml

from alphagenome_pytorch.extensions.finetuning import args as args_mod

DOCS = Path(__file__).resolve().parents[2] / "docs" / "finetuning" / "cli.rst"

#: Documented keys that are deliberately not plain scalar merges. Each needs a
#: reason, so adding to this set is a conscious act rather than a silent escape
#: hatch.
NON_SCALAR_KEYS = {
    # Nested mapping handled by its own parser.
    "modalities": "parsed per-modality, not a scalar",
    # Boolean aliases resolved explicitly after the scalar loop.
    "use_amp": "alias for the inverted --no-amp",
    "no_amp": "alias handled next to use_amp",
    "no_cache": "expands to cache_genome + cache_signals",
    "gene_expr_eval": "coerced to bool explicitly",
    # Config-settable, but per-modality rather than top-level; --strand-pairs
    # overrides the per-modality value.
    "strand_pairs": "set as modalities.<name>.strand_pairs, not a top-level scalar",
}


def _documented_top_level_keys() -> set[str]:
    """Top-level keys from the YAML schema block in the CLI docs."""
    text = DOCS.read_text()
    # The schema lives in a code-block inside the "Full Config Schema" dropdown.
    match = re.search(
        r"Full Config Schema.*?\.\. code-block:: yaml\n\n(.*?)(?=\n\S|\Z)",
        text,
        re.S,
    )
    assert match, f"Could not locate the YAML schema block in {DOCS}"
    block = match.group(1)

    # Strip the uniform RST indentation so the result is parseable YAML.
    lines = [ln for ln in block.splitlines() if ln.strip()]
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    dedented = "\n".join(ln[indent:] if len(ln) >= indent else ln
                         for ln in block.splitlines())

    data = yaml.safe_load(dedented)
    assert isinstance(data, dict), "Schema block did not parse as a mapping"
    return set(data.keys())


def _mergeable_keys() -> set[str]:
    """The scalar keys ``postprocess_args`` actually copies from the config."""
    source = Path(args_mod.__file__).read_text()
    match = re.search(r"for attr in \(\n(.*?)\n    \):", source, re.S)
    assert match, "Could not locate the scalar merge tuple in args.py"
    return set(re.findall(r'"([a-z_0-9]+)"', match.group(1)))


class TestConfigSchemaDrift:
    def test_documented_keys_are_merged(self):
        documented = _documented_top_level_keys()
        mergeable = _mergeable_keys()

        unmerged = sorted(documented - mergeable - set(NON_SCALAR_KEYS))
        assert not unmerged, (
            f"These keys are documented in {DOCS.name} but are not merged from "
            f"the YAML config, so setting them silently does nothing: {unmerged}. "
            "Either add them to the scalar merge tuple in args.py, or record why "
            "they are special in NON_SCALAR_KEYS."
        )

    def test_exemptions_are_still_special_cased(self):
        """A stale exemption would mask real drift, so keep the list honest.

        Each exempted key must still be absent from the scalar merge tuple; if
        one has since become an ordinary scalar merge, the exemption is dead
        weight and should be deleted.
        """
        mergeable = _mergeable_keys()
        stale = sorted(k for k in NON_SCALAR_KEYS if k in mergeable)
        assert not stale, (
            f"These keys are now ordinary scalar merges, so their entries in "
            f"NON_SCALAR_KEYS are stale and should be removed: {stale}"
        )

    @pytest.mark.parametrize("key", ["save_delta", "no_full_checkpoint"])
    def test_regression_keys_are_merged(self, key):
        """The two keys the original bug affected, pinned by name."""
        assert key in _mergeable_keys()
