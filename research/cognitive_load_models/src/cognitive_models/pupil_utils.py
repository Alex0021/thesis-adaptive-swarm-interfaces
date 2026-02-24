import numpy as np
import pandas as pd
import pywt

EPSILON = 1e-10


def lhipa(eye_df: pd.DataFrame, wavelet_type: str = "sym16") -> float:
    """
    Computes the LHIPA (Low-High Index of Pupillary Activity) for a given eye-tracking DataFrame.

    :param eye_df: Should have a 'pupil_diameter_mm' column with the pupil diameter measurements.
    :param wavelet_type: The type of wavelet to use for decomposition.
    :return lhipa_value: The computed LHIPA value.
    """
    data = eye_df["pupil_diameter_mm"].to_numpy().copy()
    w = pywt.Wavelet(wavelet_type)
    max_level = pywt.dwt_max_level(len(data), w.dec_len)

    hif, lof = 1, int(max_level / 2)

    cD_H = pywt.downcoef("d", data, w, level=hif, mode="per")
    cD_L = pywt.downcoef("d", data, w, level=lof, mode="per")

    # Normalize
    cD_H /= np.sqrt(2**hif)
    cD_L /= np.sqrt(2**lof)

    # Check for zero values in cD_H to avoid division by zero
    cD_H[cD_H == 0.0] = EPSILON

    cD_LH = cD_L / cD_H[[i for i in range(len(cD_H)) if i % (2 ** (lof - hif)) == 0]]

    # Modmax
    cD_LHm = modmax(cD_LH)

    # Universal threshold for noise estimation
    lambda_univ = np.std(cD_LHm) * np.sqrt(2.0 * np.log2(len(cD_LHm)))
    cD_LHt = pywt.threshold(cD_LHm, lambda_univ, mode="less")

    duration = eye_df["timestamp_sec"].max() - eye_df["timestamp_sec"].min()

    return (cD_LHt > 0).sum() / duration


def modmax(coeffs):
    abs_coeffs = np.abs(np.array(coeffs))
    lcoeffs = np.roll(abs_coeffs, -1)
    lcoeffs[-1] = 0.0
    rcoeffs = np.roll(abs_coeffs, 1)
    rcoeffs[0] = 0.0
    return np.multiply(
        abs_coeffs,
        ((abs_coeffs >= lcoeffs) & (abs_coeffs >= rcoeffs))
        & ((abs_coeffs > lcoeffs) | (abs_coeffs > rcoeffs)),
    )
