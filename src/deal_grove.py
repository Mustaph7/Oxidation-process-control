# Mustaf Bare
# Oxidation process control - I aim to predict how thick an SiO2 layer grows on a silicon wafer in a furnace, given the temperature, time, ambient (dry O2 or wet/steam) and the wafer's crystal orientation.



"""
Deal-Grove thermal oxidation model for silicon.
The model:  x^2 + A*x = B*(t + tau)
 
    x    = oxide thickness
    t    = oxidation time
    B    = parabolic rate constant (um^2/hr). Controls growth once the oxide
           is thick and oxygen has to diffuse through it (diffusion-limited).
    B/A  = linear rate constant (um/hr). Controls growth while the oxide is
           thin and the reaction at the silicon surface is the bottleneck
           (reaction-limited).
    tau  = time shift that accounts for oxide already on the wafer
 
Both B and B/A rise exponentially with temperature (Arrhenius):
    k = C * exp(-Ea / (k_B * T))
 
Constants: standard Deal-Grove values for <111> silicon at ~1 atm
(Deal & Grove, J. Appl. Phys. 36, 3770 (1965); tabulated in Jaeger,
"Introduction to Microelectronic Fabrication"). For <100> silicon the
linear rate constant B/A is divided by 1.68.
 
Units: the constants use micrometers (um) and hours (hr). Public functions
take and return thickness in nanometers (nm).
 
Known limitation: Deal-Grove underpredicts growth for very thin dry oxides
(below ~25 nm), because early growth is faster than the model assumes.
Massoud et al. (1985) give a thin-oxide correction.
"""



from pathlib import Path
import numpy as np

# Boltzmann constant in eV/K. Used in the Arrhenius equation so the activation energies can stay in electron-volts.
K_BOLTZMANN_EV = 8.617e-5  


# Deal-Grove constants for <111> silicon.
# Each entry is (pre-factor C, activation energy Ea in eV).
# Wet oxidation is much faster than dry because H2O dissolves in and diffuses through SiO2 far more easily than O2 does.
DG_CONSTANTS = {
    "dry": {
        "B": (7.72e2, 1.23),      
        "B_over_A": (6.23e6, 2.00),  
    },
    "wet": {
        "B": (3.86e2, 0.78),      
        "B_over_A": (1.63e8, 2.05),  
    },
}

# Orientation only affects the surface reaction (B/A), not diffusion (B).
# <100> silicon has fewer bonds available at the surface than <111>,
# so it reacts more slowly. Most production wafers are <100>.
ORIENTATION_FACTOR = {"111": 1.0, "100": 1.0 / 1.68}

def _arrhenius(prefactor, ea_ev, temp_c):
    temp_k = temp_c + 273.15
    return prefactor * np.exp(-ea_ev / (K_BOLTZMANN_EV * temp_k))



def rate_constants(temp_c, ambient="dry", orientation="100"):
    """
    Return the Deal-Grove rate constants at a given temperature.
 
    Returns (B, B_over_A, A):
        B         parabolic rate constant (um^2/hr)
        B_over_A  linear rate constant (um/hr)
        A         B divided by B/A (um), used in the growth equation
    """
    ambient = ambient.lower()
    if ambient not in DG_CONSTANTS:
        raise ValueError(f"ambient must be 'dry' or 'wet', got {ambient!r}")
    if orientation not in ORIENTATION_FACTOR:
        raise ValueError(f"orientation must be '100' or '111', got {orientation!r}")

    c = DG_CONSTANTS[ambient]
    B = _arrhenius(*c["B"], temp_c)
    B_over_A = _arrhenius(*c["B_over_A"], temp_c) * ORIENTATION_FACTOR[orientation]
    A = B / B_over_A
    return B, B_over_A, A


def thickness(time_hr, temp_c, ambient="dry", orientation="100", x_initial_nm=0.0):
    """
    Oxide thickness (nm) after oxidizing for time_hr hours.
 
    time_hr can be a single number or a numpy array of times.
    x_initial_nm is any oxide already on the wafer (native oxide, earlier step).
    """
    B, _, A = rate_constants(temp_c, ambient, orientation)
    x_i = x_initial_nm / 1000.0 
    tau = (x_i**2 + A * x_i) / B
    t = np.asarray(time_hr, dtype=float)

    # Solve x^2 + A x - B (t + tau) = 0 for the positive root
    x_um = (A / 2.0) * (np.sqrt(1.0 + 4.0 * B * (t + tau) / A**2) - 1.0)
    return x_um * 1000.0  


def time_to_target(target_nm, temp_c, ambient="dry", orientation="100", x_initial_nm=0.0):
   
    if target_nm <= x_initial_nm:
        return 0.0
    B, _, A = rate_constants(temp_c, ambient, orientation)
    x = target_nm / 1000.0
    x_i = x_initial_nm / 1000.0
    tau = (x_i**2 + A * x_i) / B
    return (x**2 + A * x) / B - tau


def plot_growth_curves(save_path=None):
    import matplotlib.pyplot as plt
    temps = [900, 1000, 1100]
    times = np.linspace(0, 4, 200) 

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ambient in zip(axes, ["dry", "wet"]):
        for T in temps:
            ax.plot(times, thickness(times, T, ambient), label=f"{T} °C")
        ax.set_title(f"{ambient.capitalize()} oxidation, <100> Si")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Oxide thickness (nm)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Deal-Grove thermal oxide growth")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
   
    print("Sanity check, <100> Si, bare wafer:")
    for ambient in ["dry", "wet"]:
        for T in [900, 1000, 1100]:
            x = thickness(1.0, T, ambient)
            print(f"  {ambient:>3}  {T} °C  1 hr  ->  {x:7.1f} nm")

    t = time_to_target(100, 1000, "dry")
    print(f"\nTime to grow 100 nm dry at 1000 °C: {t * 60:.1f} min")

    repo_root = Path(__file__).resolve().parents[1]
    plot_growth_curves(repo_root / "plots" / "growth_curves.png")