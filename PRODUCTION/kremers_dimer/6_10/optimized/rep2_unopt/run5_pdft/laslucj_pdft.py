from pyscf.fci.selected_ci import _as_SCIvector, make_rdm1s, make_rdm2
from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
import numpy as np

def rdms_from_sqd_sci_state(sci_state):
    sci_vec = _as_SCIvector(
        sci_state.amplitudes,
        (sci_state.ci_strs_a, sci_state.ci_strs_b),
    )
    norb = sci_state.norb
    nelec = tuple(sci_state.nelec)

    casdm1s = np.asarray(make_rdm1s(sci_vec, norb, nelec))
    casdm2 = np.asarray(make_rdm2(sci_vec, norb, nelec))

    return casdm1s, casdm2


def laslucj_sqd_pdft_energy(las, sci_state, ot="tPBE"):
    casdm1s, casdm2 = rdms_from_sqd_sci_state(sci_state)

    sqdpdft = get_mcpdft_child_class(las, ot=ot)

    def make_one_casdm1s(ci=None, state=0, **kwargs):
        return casdm1s

    def make_one_casdm2(ci=None, state=0, **kwargs):
        return casdm2

    sqdpdft.make_one_casdm1s = make_one_casdm1s
    sqdpdft.make_one_casdm2 = make_one_casdm2

    return sqdpdft.compute_pdft_energy_(las.mo_coeff)
