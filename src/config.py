# config.py
#
# Every setting for the virtual furnace lives here, with where each
# number came from. Anything marked ILLUSTRATIVE is a round value I picked
# to behave like a real tool. It is NOT taken from any production tool,
# recipe or spec.
#
# Public sources:
#   [ASM]  ASM International, A400 DUO product page
#          https://www.asm.com/our-technology-products/vertical-furnace/a400-duo
#          (200 mm, dual-reactor vertical batch furnace; oxidation, diffusion, LPCVD)
#   [SST]  "Thermal Processing: Status and future of batch, hot-wall furnaces,"
#          Solid State Technology. https://sst.semiconductor-digest.com/issue/?id=18747
#          (~100 product wafers per load, 5 heater control zones)


# ---- Tool ----------------------------------------------------------------

WAFER_DIAMETER_MM = 200        # [ASM]
REACTORS = ["A", "B"]          # [ASM] dual reactor


# ---- Boat ----------------------------------------------------------------
# Slot 1 is the top of the boat, slot 110 is the bottom.
# Real boats keep dummy wafers at both ends so the product wafers see a
# steady thermal and gas environment. ~100 product wafers per load [SST].

BOAT_SLOTS = 110
DUMMY_SLOTS = list(range(1, 6)) + list(range(106, 111))   # 5 at each end, ILLUSTRATIVE

# One monitor wafer in each heater zone. These are the only wafers that get
# measured, the same as on a real tool.
MONITOR_SLOTS = [8, 30, 55, 80, 103]                       # ILLUSTRATIVE


# ---- Heater zones --------------------------------------------------------
# 5 zones stacked top to bottom [SST], 22 slots each.
# Zone 1 = top, zone 5 = bottom.

N_ZONES = 5
SLOTS_PER_ZONE = BOAT_SLOTS // N_ZONES

# How far each zone's real temperature sits from its setpoint (°C).
# This is the "hidden truth" the zone tuning has to figure out from monitor
# data alone. Top and bottom run cooler because the ends of the tube lose
# heat. Reactor B is offset differently from A on purpose, so the two
# reactors don't match out of the box (Case 3).
TRUE_ZONE_OFFSETS_C = {                                    # ILLUSTRATIVE
    "A": [-1.5, 0.4, 0.0, 0.3, -1.0],
    "B": [-0.5, 1.3, 0.9, 1.0, -0.6],
}


# ---- Gas flow ------------------------------------------------------------
# Gas enters at the bottom and flows up the boat. A little oxidant gets used
# up along the way, so wafers near the top see slightly less of it.
# DEPLETION is the fraction lost by the time the gas reaches the top slot.

GAS_INLET = "bottom"
DEPLETION = 0.01                                           # ILLUSTRATIVE (1%)


# ---- Noise ---------------------------------------------------------------

RUN_TEMP_NOISE_C = 0.3       # whole-furnace temperature wobble, run to run (°C, 1 sigma)
WAFER_NOISE_FRAC = 0.0015    # random wafer-to-wafer thickness scatter (0.15%, 1 sigma)
METROLOGY_NOISE_A = 1.0      # thickness measurement repeatability (Å, 1 sigma)
WIW_EDGE_FRAC = 0.004        # wafer edge grows 0.4% thicker than center
# all ILLUSTRATIVE


# ---- Measurement pattern -------------------------------------------------
# 9-site pattern on each monitor wafer: center, 4 sites at half radius,
# 4 sites near the edge. Each site is (radius as fraction of wafer radius, angle in degrees).

SITES = [
    (0.0, 0),
    (0.5, 0), (0.5, 90), (0.5, 180), (0.5, 270),
    (0.9, 45), (0.9, 135), (0.9, 225), (0.9, 315),
]


# ---- Baseline recipe -----------------------------------------------------
# Dry oxidation targeting 1000 Å (100 nm). deal_grove.time_to_target says
# 100 nm dry at 1000 °C takes ~191 min, so 3.2 hr lands right around it.

BASE_RECIPE = {
    "ambient": "dry",
    "temp_c": 1000.0,
    "time_hr": 3.2,
}
TARGET_A = 1000.0