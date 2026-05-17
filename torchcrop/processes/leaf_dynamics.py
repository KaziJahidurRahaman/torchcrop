"""Leaf area growth and senescence.

Ports the Lintul5 ``GLA`` (Growth Leaf Area) and ``DEATHL`` (leaf death)
subroutines from ``LintulFunctions.java`` together with the ``SLA`` and
``GLAI``/``DLAI`` block of ``Lintul5.java``.

References:
    * ``simplace/sim/components/models/lintul5/LintulFunctions.java`` —
      ``GLA`` (lines 998–1021) and ``DEATHL``.
    * ``simplace/sim/components/models/lintul5/Lintul5.java`` — the
      ``process()`` block that wires SLA, GLAI, DLAI together.

Three growth regimes (precedence: emergence > juvenile > mature):

* **Emergence day** (``LAI == 0``): ``GLAI = LAII / DELT``.
* **Juvenile** (``DVS < 0.2`` *and* ``LAI < 0.75``): exponential
    ``GLAI = LAI · (exp(RGRLAI · DTEFF) − 1) · TRANRF · exp(−NLAI · (1−NPKI))``.
* **Mature**: source-limited ``GLAI = SLA · GLV``.

Senescence aggregates three independent death drivers via ``max``:
ageing/temperature (``RDRTMP`` indexed by **mean air temperature**, gated
by ``DVSDLT``), self-shading above ``LAICR``, and drought
``(1−TRANRF)·RDRL``. Heat stress multiplies the resulting ``RDR`` and the
result is capped at ``1`` d⁻¹. NPK-driven senescence is **additive**:
``DLVNS = WLVG · RDRNS · (1−NPKI)`` (with ``DLAINS = DLVNS · SLA``), and
SLA itself carries an ``exp(−NSLA·(1−NPKI))`` reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class LeafDynamics(nn.Module):
    """Leaf area growth, senescence, dead-leaf accumulation."""

    def forward(
        self,
        state: ModelState,
        g_lv: torch.Tensor,
        dtsu: torch.Tensor,
        davtmp: torch.Tensor,
        tranrf: torch.Tensor,
        nstress: torch.Tensor,
        params: CropParameters,
        heat_stress: torch.Tensor | None = None,
        emerg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute leaf area and leaf-biomass rates for one day.

        Args:
            state: Current state; uses ``state.lai``, ``state.wlv``,
                ``state.dvs``.
            g_lv: Leaf growth allocated by partitioning [g DM m⁻² d⁻¹],
                shape ``[B]``. Equivalent to Lintul5 ``GLV``.
            dtsu: Effective thermal time for LAI growth [°C d d⁻¹], shape
                ``[B]``. Equivalent to Lintul5 ``DTEFF``.
            davtmp: Mean daily air temperature [°C], shape ``[B]``.
                Equivalent to Lintul5 ``TMPA`` — drives ``RDRTMP`` via
                interpolation on ``rdrltb``.
            tranrf: Water-stress factor in ``[0, 1]``, shape ``[B]``.
            nstress: NPK nutrient index ``NPKI`` in ``[0, 1]``, shape
                ``[B]``.
            params: Crop parameters; uses ``laicr``, ``rgrl``, ``slatb``,
                ``scale_factor_sla``, ``nsla``, ``nlai``, ``laii``,
                ``rdrshm``, ``rdrl``, ``rdrns``, ``rdrltb``,
                ``scale_factor_rdr_leaves``, ``dvsdlt``.
            heat_stress: Optional multiplicative heat-stress factor on
                ``RDR`` (Lintul5 ``iLeaveSenescenceHeatStressFactor``,
                default ``1.0``), broadcastable to ``[B]``.
            emerg: Optional emergence mask in ``{0, 1}`` (broadcast to
                ``[B]``). Matches Lintul5 ``EMERG``: when ``0``, both
                ``GLAI`` and ``DLV``/``DLAI`` are forced to zero (so
                pre-emergence leaves neither grow nor senesce). Default
                is derived from ``state.tsump >= params.tsumem``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine for state update):

                * ``lai_rate`` [m² m⁻² d⁻¹] — Net daily change in LAI
                  (``= glai − dlai``).
                * ``wlv_rate`` [g DM m⁻² d⁻¹] — Net daily change in green
                  leaf weight (``= g_lv − dlv``).
                * ``wlvd_rate`` [g DM m⁻² d⁻¹] — Daily senesced leaf mass
                  transferred into the dead-leaf pool ``wlvd``
                  (``= dlv``).

            Diagnostics:

                * ``lai_growth`` [m² m⁻² d⁻¹] — Daily LAI growth (GLA).
                * ``lai_sen`` [m² m⁻² d⁻¹] — Daily LAI loss to
                  senescence (DLAI = DLAIS + DLAINS).
                * ``rdr`` [d⁻¹] — Effective relative death rate (after
                  heat scaling and ≤ 1 cap), excluding the additive NPK
                  death.
                * ``sla`` [m² g⁻¹] — Effective specific leaf area after
                  NPK reduction.
        """
        lai = state.lai
        wlv = state.wlv
        dvs = state.dvs

        if heat_stress is None:
            heat_stress = torch.ones_like(lai)
        if emerg is None:
            emerg = (state.tsump >= params.tsumem).to(lai.dtype)
        else:
            emerg = emerg.to(lai.dtype)

        # ----- SLA with NPK reduction -----
        # Lintul5.java:1408:
        #   SLA = cScaleFactorSLA * SLATB(DVS) * exp(-NSLA * (1 - NPKI))
        sla_base = interpolate(params.slatb, dvs)
        sla = params.scale_factor_sla * sla_base * torch.exp(
            -params.nsla * (1.0 - nstress)
        )

        # ----- GLA: daily increase in leaf area index -----
        # LintulFunctions.java:998–1019. Branch precedence in the Java
        # source (last assignment wins): emergence > juvenile > mature.
        glai_mature = sla * g_lv
        glai_juv = (
            lai
            * (torch.exp(params.rgrl * dtsu) - 1.0)
            * tranrf
            * torch.exp(-params.nlai * (1.0 - nstress))
        )
        glai_emerg = torch.broadcast_to(params.laii, lai.shape)

        juv_mask = (dvs < 0.2) & (lai < 0.75)
        emerg_mask = lai <= 0.0
        glai = torch.where(
            emerg_mask,
            glai_emerg,
            torch.where(juv_mask, glai_juv, glai_mature),
        )

        # ----- DEATHL: relative death rates -----
        # LintulFunctions.java:943–948.
        rdrtmp = interpolate(params.rdrltb, davtmp) * params.scale_factor_rdr_leaves
        rdrdv = torch.where(dvs < params.dvsdlt, torch.zeros_like(rdrtmp), rdrtmp)
        rdrsh = torch.clamp(
            params.rdrshm * (lai - params.laicr) / _safe(params.laicr),
            min=0.0,
        )
        rdrdry = (1.0 - tranrf) * params.rdrl
        rdr = torch.maximum(torch.maximum(rdrdv, rdrsh), rdrdry) * heat_stress
        rdr = torch.clamp(rdr, max=1.0)

        # Senescence from drivers (max of RDRDV / RDRSH / RDRDRY).
        # LintulFunctions.java:962–969.
        dlvs = wlv * rdr
        dlais = lai * rdr

        # Additive NPK senescence. LintulFunctions.java:950–958.
        # DLVNS = WLVG * RDRNS * (1 - NPKI)  iff NPKI < 1
        # DLAINS = DLVNS * SLA  (mass→area via SLA, not LAI/WLV ratio)
        npki_deficit = torch.clamp(1.0 - nstress, min=0.0)
        dlvns = wlv * params.rdrns * npki_deficit
        dlains = dlvns * sla

        dlv = dlvs + dlvns
        dlai = dlais + dlains

        # EMERG gating — Java GLA/DEATHL return 0 when EMERG is false.
        glai = glai * emerg
        dlv = dlv * emerg
        dlai = dlai * emerg

        lai_rate = glai - dlai
        wlv_rate = g_lv - dlv

        return {
            "lai_rate": lai_rate,
            "wlv_rate": wlv_rate,
            "wlvd_rate": dlv,
            "lai_growth": glai,
            "lai_sen": dlai,
            "rdr": rdr,
            "sla": sla,
        }


def _safe(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Guard a denominator against zero.

    Args:
        t: Denominator tensor.
        eps: Threshold below which ``t`` is replaced by ``1``.

    Returns:
        ``t`` with near-zero entries replaced by ones.
    """
    return torch.where(t.abs() > eps, t, torch.ones_like(t))
