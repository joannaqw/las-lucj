from collections import OrderedDict
from itertools import product

#copy and pasted from tutorial:
def unique_alpha_beta_combined(bitstrings):
    """A utility function for getting unique alpha and beta halves from full bitstrings.

    Args:
        bitstrings (list[str]): A list of full bitstrings consisting of both alpha and beta parts.

    Returns:
        A dictionary with combined unique ``alpha`` and ``beta`` half strings.
    """
    if not bitstrings:
        return OrderedDict()

    unique_ab = OrderedDict()
    num_spatial_orb = len(bitstrings[0]) // 2

    for bs in bitstrings:
        a = bs[num_spatial_orb:]
        print('combined a',a)
        b = bs[:num_spatial_orb]
        print('combined b',b)
        unique_ab[a] = 1
        print('combined, unique a', unique_ab)
        unique_ab[b] = 1
        print('combined, unique b', unique_ab)
        print(unique_ab)
    return unique_ab

def unique_alpha_beta_separate(bitstrings):
    """Open-shell-safe: split alpha/beta and keep two separate pools."""
    alpha_dict = OrderedDict()
    beta_dict = OrderedDict()

    if not bitstrings:
        return alpha_dict, beta_dict

    norb = len(bitstrings[0]) // 2

    for bs in bitstrings:
        beta = bs[:norb]
        alpha = bs[norb:]

        alpha_dict[alpha] = 1
        beta_dict[beta] = 1
        print('alpha_dict',alpha_dict)
        print('beta_dict', beta_dict)
    return alpha_dict, beta_dict


def hamming_weight(bitstr):
    return bitstr.count("1")


def print_subspace(label, alpha_pool, beta_pool, num_alpha, num_beta):
    print(f"\n--- {label} ---")
    print("alpha pool:", alpha_pool)
    print("beta pool: ", beta_pool)
    print(f"subspace dimension = {len(alpha_pool)} x {len(beta_pool)} = {len(alpha_pool) * len(beta_pool)}")

    print("\nConfigurations generated:")
    for alpha, beta in product(alpha_pool, beta_pool):
        alpha_ok = hamming_weight(alpha) == num_alpha
        beta_ok = hamming_weight(beta) == num_beta
        valid = alpha_ok and beta_ok

        mark = "valid" if valid else "invalid"

        print(
            f"alpha={alpha} ({hamming_weight(alpha)}e), "
            f"beta={beta} ({hamming_weight(beta)}e)  {mark}"
        )


def run_example(name, bitstrings, num_alpha, num_beta):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Target: num_alpha={num_alpha}, num_beta={num_beta}")

    print("\nInput full bitstrings are beta + alpha:")
    norb = len(bitstrings[0]) // 2
    for bs in bitstrings:
        beta = bs[:norb]
        alpha = bs[norb:]
        print(f"full={bs}   beta={beta} ({hamming_weight(beta)}e), alpha={alpha} ({hamming_weight(alpha)}e)")

    # Tutorial-style combined pool
    combined = unique_alpha_beta_combined(bitstrings)
    half_strs = list(combined.keys())

    print_subspace(
        label="Tutorial combined version: Subspace([half_strs, half_strs])",
        alpha_pool=half_strs,
        beta_pool=half_strs,
        num_alpha=num_alpha,
        num_beta=num_beta,
    )

    # Separate alpha/beta pools
    alpha_dict, beta_dict = unique_alpha_beta_separate(bitstrings)
    alpha_half_strs = list(alpha_dict.keys())
    beta_half_strs = list(beta_dict.keys())

    print_subspace(
        label="Separate version: Subspace([alpha_half_strs, beta_half_strs])",
        alpha_pool=alpha_half_strs,
        beta_pool=beta_half_strs,
        num_alpha=num_alpha,
        num_beta=num_beta,
    )


if __name__ == "__main__":
    # Closed-shell / spin-balanced example
    # num_alpha = num_beta = 2
    closed_shell_bitstrings = [
        "1100" + "0011",  # beta=1100, alpha=0011
        "1010" + "0101",  # beta=1010, alpha=0101
    ]

    run_example(
        name="CLOSED-SHELL EXAMPLE",
        bitstrings=closed_shell_bitstrings,
        num_alpha=2,
        num_beta=2,
    )

    # Open-shell / spin-unbalanced example
    # num_alpha = 3, num_beta = 1
    open_shell_bitstrings = [
        "1000" + "1110",  # beta=1000, alpha=1110
        "0100" + "1101",  # beta=0100, alpha=1101
    ]

    run_example(
        name="OPEN-SHELL EXAMPLE",
        bitstrings=open_shell_bitstrings,
        num_alpha=3,
        num_beta=1,
    )
