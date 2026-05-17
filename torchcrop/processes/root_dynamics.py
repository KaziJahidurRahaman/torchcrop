"""Root depth growth, root biomass and root senescence.

Ports the Lintul5 root block from ``Lintul5.java`` together with the
``DeadRootsStemsRate`` subroutine from ``LintulFunctions.java``. Root
depth grows at the maximum daily increase ``RRI`` whenever the crop has
emerged and the soil is not severely water-stressed; growth is capped
at the maximum rooting depth ``RDM`` and gated by the SIMPLACE
``INSW``-style water-stress step (zero growth iff ``TRANRF < 0.01``).

References:
    * ``simplace/sim/components/models/lintul5/Lintul5.java`` — root
      growth rate ``RR`` (lines 1472–1474) and the Java integrations
      ``INTGRL(WRT, RWRT)`` / ``INTGRL(WRTD, DRRT)`` / ``INTGRL(RD, RR)``.
    * ``simplace/sim/components/models/lintul5/LintulFunctions.java`` —
      ``DeadRootsStemsRate`` (lines 247–269) and the ``RELGR`` root term
      ``RWRT = GRT·FRT − DRRT`` (line 1052).
    * ``simplace/sim/components/util/FSTFunctions.java`` — the
      ``INSW(testvalue, X2, X3)`` step function reproduced verbatim
      below.

Java reference snippets:
    Root depth (Lintul5.java)::

        if (EMERG)
            RR = min(RRI * INSW(TRANRF − 0.01, 0., 1.), RDM − RD);

    Root senescence (LintulFunctions.java)::

        double DRRT = 0.0;
        if (DVS >= DVSDR) DRRT = WRT * RDRRT;

    where ``RDRRT = RDRRTB(DVS) * cScaleFactorRDRRoots`` (Lintul5.java
    line 1441) is the DVS-indexed relative root death rate.

Net root biomass change therefore is ``RWRT = GRT·FRT − DRRT`` per the
``RELGR`` subroutine; in torchcrop ``GRT·FRT`` arrives pre-computed as
``g_root`` from `Partitioning`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class RootDynamics(nn.Module):
    """Root depth, root biomass and dead-root rates."""

    def forward(
        self,
        state: ModelState,
        g_root: torch.Tensor,
        tranrf: torch.Tensor,
        params: CropParameters,
        emerg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute root depth and root-biomass rates for one day.

        Args:
            state: Current state; uses ``state.rootd``, ``state.dvs``,
                ``state.wrt`` and (for the default emergence mask)
                ``state.tsump``.
            g_root: Gross root biomass growth from partitioning
                [g DM m⁻² d⁻¹], shape ``[B]`` — equivalent to the Java
                ``GRT * FRT`` term.
            tranrf: Water-stress factor in ``[0, 1]``, shape ``[B]``.
            params: Crop parameters; uses ``rri`` (maximum daily root
                depth increase), ``rdmcr`` (crop-specific maximum
                rooting depth), ``rdrrtb`` (relative root death rate vs
                DVS), ``scale_factor_rdr_roots`` and ``dvsdr`` (DVS
                threshold above which root death starts).
            emerg: Optional emergence mask in ``{0, 1}`` (broadcast to
                ``[B]``). Matches Lintul5 ``EMERG``: when ``0`` the root
                front does not advance (pre-emergence) and dead-root
                accumulation is also suppressed. Default is derived
                from ``state.tsump >= params.tsumem``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine for state update):

                * ``rootd_rate`` [m d⁻¹] — Daily increment of rooting
                  depth, ``min(RRI · INSW(TRANRF − 0.01, 0, 1),
                  RDM − RD)``, gated to zero pre-emergence.
                * ``wrt_rate`` [g DM m⁻² d⁻¹] — Net daily change in
                  living root biomass (``= g_root − drrt``).
                * ``wrtd_rate`` [g DM m⁻² d⁻¹] — Daily senesced root
                  mass transferred to the dead-root pool (``= drrt``).

            Diagnostics:

                * ``drrt`` [g DM m⁻² d⁻¹] — Root death rate
                  ``DRRT = WRT · RDRRT · 𝟙[DVS ≥ DVSDR]``.
                * ``rdrrt`` [d⁻¹] — Effective DVS-indexed relative
                  root death rate after the scale factor.
        """
        rootd = state.rootd
        dvs = state.dvs
        wrt = state.wrt

        dtype = rootd.dtype
        zero = torch.zeros_like(rootd)
        one = torch.ones_like(rootd)

        if emerg is None:
            emerg = (state.tsump >= params.tsumem).to(dtype)
        else:
            emerg = emerg.to(dtype)

        # ----- Root depth growth -----
        # FSTFunctions.INSW(x, X2, X3) returns X2 if x < 0, else X3.
        # Lintul5.java:1474 :
        #   RR = min( RRI * INSW(TRANRF - 0.01, 0., 1.), RDM - RD )
        # i.e. a hard step at TRANRF = 0.01 — full root-front velocity
        # above the threshold, zero below — *not* a linear scaling.
        insw_water = torch.where(tranrf - 0.01 < 0.0, zero, one)
        rr_pot = params.rri * insw_water

        # Avoid overshoot beyond the crop-specific maximum rooting depth
        # (Lintul5.java uses min(..., RDM - RD); RDM is the maximum
        # rooting depth honoured by the soil/profile, capped here by the
        # crop's rdmcr parameter).
        headroom = torch.clamp(params.rdmcr - rootd, min=0.0)
        rootd_rate = torch.minimum(rr_pot, headroom)

        # Lintul5.java:1473 — root growth only after emergence.
        rootd_rate = rootd_rate * emerg

        # ----- Root senescence (DRRT) -----
        # LintulFunctions.DeadRootsStemsRate:
        #   DRRT = WRT * RDRRT   if DVS >= DVSDR else 0
        # RDRRT = RDRRTB(DVS) * cScaleFactorRDRRoots   (Lintul5.java:1441)
        rdrrt = interpolate(params.rdrrtb, dvs) * params.scale_factor_rdr_roots
        death_mask = (dvs >= params.dvsdr).to(dtype)
        drrt = wrt * rdrrt * death_mask * emerg

        # ----- Net living-root biomass change (RELGR root term) -----
        # LintulFunctions.RELGR:  RWRT = GRT * FRT - DRRT
        # g_root already carries the GRT * FRT contribution from
        # `Partitioning`, so we just subtract the senescence flux.
        wrt_rate = g_root - drrt

        return {
            "rootd_rate": rootd_rate,
            "wrt_rate": wrt_rate,
            "wrtd_rate": drrt,
            "drrt": drrt,
            "rdrrt": rdrrt,
        }
