"""Downstream forecasting backbones."""

from . import patchstg, stid

MODELS = {"STID": stid, "PatchSTG": patchstg}
