"""Stem biomass net growth and stem senescence.

Ports the stem half of the SIMPLACE ``DeadRootsStemsRate`` subroutine
(``LintulFunctions.java``) together with the stem branch of ``RELGR``
(``RWST = AGRT·FST − DRST``). Stem senescence shares the ``DVS ≥ DVSDR``
gate with root senescence but uses its own DVS-indexed relative death
rate table ``RDRSTB``.

References:
    * ``simplace/sim/components/models/lintul5/Lintul5.java`` —
      ``RDRST = RDRSTB(DVS) * cScaleFactorRDRStems`` (line 1442),
      ``INTGRL(WST, RWST)`` (line 1094) and ``INTGRL(WSTD, DRST)``
      (line 1102).
    * ``simplace/sim/components/models/lintul5/LintulFunctions.java`` —
      ``DeadRootsStemsRate`` (lines 247–269) and the ``RELGR`` stem
      term ``RWST = AGRT·FST − DRST`` (line 1055).

Java reference snippets:
    Stem senescence (LintulFunctions.java)::

        double DRST = 0.0;
        if (DVS >= DVSDR) DRST = WST * RDRST;

    Net stem biomass change (LintulFunctions.java:1055)::

        RWST = AGRT * FST - DRST;

In torchcrop the ``AGRT · FST`` term arrives pre-computed as ``g_st``
from `Partitioning`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class StemDynamics(nn.Module):
    """Stem biomass net rate and dead-stem accumulation rate."""

    def forward(
        self,
        state: ModelState,
        g_st: torch.Tensor,
        params: CropParameters,
        emerg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute stem biomass rates for one day.

        Args:
            state: Current state; uses ``state.dvs``, ``state.wst`` and
                (for the default emergence mask) ``state.tsump``.
            g_st: Gross stem biomass growth from partitioning
                [g DM m⁻² d⁻¹], shape ``[B]`` — equivalent to the Java
                ``AGRT * FST`` term.
            params: Crop parameters; uses ``rdrstb`` (relative stem
                death rate vs DVS), ``scale_factor_rdr_stems`` and
                ``dvsdr`` (DVS threshold above which stem death starts).
            emerg: Optional emergence mask in ``{0, 1}`` (broadcast to
                ``[B]``). Matches Lintul5 ``EMERG``: when ``0`` dead
                stems do not accumulate. Default is derived from
                ``state.tsump >= params.tsumem``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine for state update):

                * ``wst_rate`` [g DM m⁻² d⁻¹] — Net daily change in
                  living stem biomass (``= g_st − drst``).
                * ``wstd_rate`` [g DM m⁻² d⁻¹] — Daily senesced stem
                  mass transferred to the dead-stem pool (``= drst``).

            Diagnostics:

                * ``drst`` [g DM m⁻² d⁻¹] — Stem death rate
                  ``DRST = WST · RDRST · 𝟙[DVS ≥ DVSDR]``.
                * ``rdrst`` [d⁻¹] — Effective DVS-indexed relative
                  stem death rate after the scale factor.
        """
        dvs = state.dvs
        wst = state.wst
        dtype = wst.dtype

        if emerg is None:
            emerg = (state.tsump >= params.tsumem).to(dtype)
        else:
            emerg = emerg.to(dtype)

        # ----- Stem senescence (DRST) -----
        # LintulFunctions.DeadRootsStemsRate:
        #   DRST = WST * RDRST   if DVS >= DVSDR else 0
        # RDRST = RDRSTB(DVS) * cScaleFactorRDRStems   (Lintul5.java:1442)
        rdrst = interpolate(params.rdrstb, dvs) * params.scale_factor_rdr_stems
        death_mask = (dvs >= params.dvsdr).to(dtype)
        drst = wst * rdrst * death_mask * emerg

        # ----- Net living-stem biomass change (RELGR stem term) -----
        # LintulFunctions.RELGR:  RWST = AGRT * FST - DRST
        wst_rate = g_st - drst

        return {
            "wst_rate": wst_rate,
            "wstd_rate": drst,
            "drst": drst,
            "rdrst": rdrst,
        }
