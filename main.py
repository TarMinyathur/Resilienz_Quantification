import pandapower.networks as pn
import networkx as nx
import numpy as np
import pandas as pd
import time
from count_elements import count_elements
from diversity import calculate_shannon_evenness_and_variety
from disparity import calculate_disparity_space, calculate_line_disparity, calculate_transformer_disparity, \
    calculate_load_disparity
from GenerationFactors import calculate_generation_factors
from Redundancy_new import Redundancy
from Redundancy import n_3_redundancy_check
from visualize import plot_spider_chart
from initialize import add_indicator
from initialize import add_disparity
from indi_gt import GraphenTheorieIndicator
from adjustments_new import set_missing_limits
from adjustments_new import determine_minimum_ext_grid_power
from self_sufficiency import selfsuff
from self_sufficiency import selfsufficiency_neu
from flexibility import calculate_flexibility
from buffer import calculate_buffer
from fxor import flexibility_fxor
from stressors import stress_scenarios
from Evaluate_scenario import run_scenario
import os
import pandapower.converter as pc
import simbench as sb
import pandapower as pp

# Dictionary to including all grid names to functions, including special cases for test grids, whose opp converges
grids = {
    # "GBreducednetwork": pn.GBreducednetwork,
    # "case118": pn.case118,
    # "case14": pn.case14,
    # "case24_ieee_rts": pn.case24_ieee_rts,
    # "case30": pn.case30,
    # "case33bw": pn.case33bw,
    # "case39": pn.case39,
    # "case5": pn.case5,
    # "case6ww": pn.case6ww,
    # "case9": pn.case9,
    # "create_cigre_network_lv": pn.create_cigre_network_lv,
    # #"create_cigre_network_mv": pn.create_cigre_network_mv,
     "create_cigre_network_mv_all": lambda: pn.create_cigre_network_mv(with_der="all"),
    # # #"create_cigre_network_mv_pv_wind": lambda: pn.create_cigre_network_mv(with_der="pv_wind"),
    # "ieee_european_lv_asymmetric": pn.ieee_european_lv_asymmetric,
    # #
    # # # Special Cases with Adjustments
    # # # "mv_all_high10": lambda: increase_generation(pn.create_cigre_network_mv(with_der="all"), factor=10),
    # # # "mv_all_high5": lambda: increase_generation(pn.create_cigre_network_mv(with_der="all"), factor=5),
    # #
    # # "example_multivoltage": lambda: increase_line_limits(pn.example_multivoltage(), 1.5),
    # # "example_simple": lambda: increase_line_limits(pn.example_simple(), 1.5),
    # # "mv_oberrhein": lambda: increase_line_limits(pn.mv_oberrhein(), 1.5),
    # #
    # # # # High-voltage grids
    #"1-HV-mixed--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-HV-mixed--0-sw"), 1.5),
    # # "1-HV-mixed--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-HV-mixed--1-sw"), 1.5),
    # # "1-HV-urban--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-HV-urban--0-sw"), 1.5),
    # # "1-HV-urban--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-HV-urban--1-sw"), 1.5),
    # # #
    # # # # Low-voltage grids
    # # "1-LV-rural1--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural1--0-sw"), 1.5),
    # # "1-LV-rural2--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural2--0-sw"), 1.5),
    # # "1-LV-rural2--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural2--1-sw"), 1.5),
    # # "1-LV-rural2--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural2--2-sw"), 1.5),
    # # "1-LV-rural3--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural3--0-sw"), 1.5),
    # # "1-LV-rural3--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural3--1-sw"), 1.5),
    # # "1-LV-rural3--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-rural3--2-sw"), 1.5),
    # # "1-LV-semiurb4--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb4--0-sw"), 1.5),
    # # "1-LV-semiurb4--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb4--1-sw"), 1.5),
    # # "1-LV-semiurb4--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb4--2-sw"), 1.5),
    # # "1-LV-semiurb5--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb5--0-sw"), 1.5),
    # # "1-LV-semiurb5--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb5--1-sw"), 1.5),
    # # "1-LV-semiurb5--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-semiurb5--2-sw"), 1.5),
    # # "1-LV-urban6--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-urban6--0-sw"), 1.5),
    # # "1-LV-urban6--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-urban6--1-sw"), 1.5),
    # # "1-LV-urban6--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-LV-urban6--2-sw"), 1.5),
    # #
    # # # Medium-voltage grids (not already added)
    # "1-MV-comm--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-comm--0-sw"), 1.5),
    # # "1-MV-comm--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-comm--1-sw"), 1.5),
    # # "1-MV-comm--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-comm--2-sw"), 1.5),
    # # "1-MV-rural--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-rural--0-sw"), 1.5),
    # # "1-MV-rural--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-rural--1-sw"), 1.5),
    # # "1-MV-rural--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-rural--2-sw"), 1.5),
    # # "1-MV-semiurb--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-semiurb--0-sw"), 1.5),
    # # "1-MV-semiurb--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-semiurb--1-sw"), 1.5),
    # # "1-MV-semiurb--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-semiurb--2-sw"), 1.5),
    # # "1-MV-urban--0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-urban--0-sw"), 1.5),
    # # "1-MV-urban--1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-urban--1-sw"), 1.5),
    # # "1-MV-urban--2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MV-urban--2-sw"), 1.5),
    # #
    # # # -- HVMV-mixed grids --
    # # "1-HVMV-mixed-1.105-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-1.105-0-no_sw"), 1.5),
    # # "1-HVMV-mixed-1.105-0-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-1.105-0-sw"), 1.5),
    # # "1-HVMV-mixed-1.105-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-1.105-1-no_sw"), 1.5),
    # # "1-HVMV-mixed-1.105-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-1.105-1-sw"), 1.5),
    # # "1-HVMV-mixed-2.102-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-2.102-1-no_sw"), 1.5),
    # # "1-HVMV-mixed-2.102-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-2.102-1-sw"), 1.5),
    # # "1-HVMV-mixed-4.101-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-4.101-1-no_sw"), 1.5),
    # # "1-HVMV-mixed-4.101-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-4.101-1-sw"), 1.5),
    # # # "1-HVMV-mixed-all-1-no_sw":   lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-all-1-no_sw"), 1.5),
    # # # "1-HVMV-mixed-all-1-sw":      lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-mixed-all-1-sw"), 1.5),
    #
    # # # -- HVMV-urban grids --
    # # "1-HVMV-urban-2.203-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-2.203-0-no_sw"), 1.5),
    # # "1-HVMV-urban-2.203-0-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-2.203-0-sw"), 1.5),
    # # "1-HVMV-urban-2.203-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-2.203-1-no_sw"), 1.5),
    # # "1-HVMV-urban-2.203-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-2.203-1-sw"), 1.5),
    # # "1-HVMV-urban-3.201-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-3.201-0-no_sw"), 1.5),
    # # "1-HVMV-urban-3.201-0-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-3.201-0-sw"), 1.5),
    # # "1-HVMV-urban-3.201-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-3.201-1-no_sw"), 1.5),
    # # "1-HVMV-urban-3.201-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-3.201-1-sw"), 1.5),
    # # "1-HVMV-urban-4.201-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-4.201-0-no_sw"), 1.5),
    # # "1-HVMV-urban-4.201-0-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-4.201-0-sw"), 1.5),
    # # "1-HVMV-urban-4.201-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-4.201-1-no_sw"), 1.5),
    # # "1-HVMV-urban-4.201-1-sw":    lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-4.201-1-sw"), 1.5),
    # # # "1-HVMV-urban-all-0-no_sw":   lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-all-0-no_sw"), 1.5),
    # # # "1-HVMV-urban-all-0-sw":      lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-all-0-sw"), 1.5),
    # # # "1-HVMV-urban-all-1-no_sw":   lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-all-1-no_sw"), 1.5),
    # # # "1-HVMV-urban-all-1-sw":      lambda: increase_line_limits(sb.get_simbench_net("1-HVMV-urban-all-1-sw"), 1.5),
    # # #
    # # # # MVLV grids – Combined medium and low voltage
    # #  "1-MVLV-comm-3.403-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-0-no_sw"), 1.5),
    # # # #"1-MVLV-comm-3.403-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-0-sw"), 1.5),
    # # "1-MVLV-comm-3.403-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-1-no_sw"), 1.5),
    # # # #"1-MVLV-comm-3.403-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-1-sw"), 1.5),
    # # "1-MVLV-comm-3.403-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-2-no_sw"), 1.5),
    # # # #"1-MVLV-comm-3.403-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-3.403-2-sw"), 1.5),
    # # "1-MVLV-comm-4.416-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-0-no_sw"), 1.5),
    # # # #"1-MVLV-comm-4.416-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-0-sw"), 1.5),
    # # "1-MVLV-comm-4.416-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-1-no_sw"), 1.5),
    # # # #"1-MVLV-comm-4.416-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-1-sw"), 1.5),
    # # "1-MVLV-comm-4.416-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-2-no_sw"), 1.5),
    # # # #"1-MVLV-comm-4.416-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-4.416-2-sw"), 1.5),
    # # "1-MVLV-comm-5.401-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-0-no_sw"), 1.5),
    # # # #"1-MVLV-comm-5.401-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-0-sw"), 1.5),
    # # "1-MVLV-comm-5.401-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-1-no_sw"), 1.5),
    # # # #"1-MVLV-comm-5.401-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-1-sw"), 1.5),
    # # "1-MVLV-comm-5.401-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-2-no_sw"), 1.5),
    # # # #"1-MVLV-comm-5.401-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-5.401-2-sw"), 1.5),
    # # #"1-MVLV-comm-all-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-all-0-no_sw"), 1.5),
    # # # #"1-MVLV-comm-all-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-comm-all-0-sw"), 1.5),
    # # #
    # # "1-MVLV-rural-1.108-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-1.108-0-no_sw"), 1.5),
    # # "1-MVLV-rural-1.108-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-1.108-0-sw"), 1.5),
    # #
    # "1-MVLV-rural-2.107-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-0-no_sw"), 1.5),
    # # #"1-MVLV-rural-2.107-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-0-sw"), 1.5),
    # "1-MVLV-rural-2.107-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-1-no_sw"), 1.5),
    # # #"1-MVLV-rural-2.107-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-1-sw"), 1.5),
    # "1-MVLV-rural-2.107-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-2-no_sw"), 1.5),
    # # #"1-MVLV-rural-2.107-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-2.107-2-sw"), 1.5),
    # "1-MVLV-rural-4.101-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-0-no_sw"), 1.5),
    # # #"1-MVLV-rural-4.101-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-0-sw"), 1.5),
    # "1-MVLV-rural-4.101-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-1-no_sw"), 1.5),
    # # #"1-MVLV-rural-4.101-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-1-sw"), 1.5),
    # "1-MVLV-rural-4.101-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-2-no_sw"), 1.5),
    # #"1-MVLV-rural-4.101-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-4.101-2-sw"), 1.5),
    # "1-MVLV-rural-all-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-all-0-no_sw"), 1.5),
    # #"1-MVLV-rural-all-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-rural-all-0-sw"), 1.5),
    #
    # "1-MVLV-semiurb-3.202-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-0-no_sw"),1.5),
    # # #"1-MVLV-semiurb-3.202-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-0-sw"), 1.5),
    # "1-MVLV-semiurb-3.202-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-1-no_sw"), 1.5),
    # #"1-MVLV-semiurb-3.202-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-1-sw"), 1.5),
    # "1-MVLV-semiurb-3.202-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-2-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-3.202-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-3.202-2-sw"), 1.5),
    # "1-MVLV-semiurb-4.201-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-0-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-4.201-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-0-sw"), 1.5),
    # "1-MVLV-semiurb-4.201-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-1-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-4.201-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-1-sw"), 1.5),
    # "1-MVLV-semiurb-4.201-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-2-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-4.201-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-4.201-2-sw"), 1.5),
    # "1-MVLV-semiurb-5.220-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-0-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-5.220-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-0-sw"), 1.5),
    # "1-MVLV-semiurb-5.220-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-1-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-5.220-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-1-sw"), 1.5),
    # "1-MVLV-semiurb-5.220-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-2-no_sw"),
    #                                                              1.5),
    # #"1-MVLV-semiurb-5.220-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-5.220-2-sw"), 1.5),
    # #"1-MVLV-semiurb-all-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-all-0-no_sw"), 1.5),
    # #"1-MVLV-semiurb-all-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-semiurb-all-0-sw"), 1.5),
    # #
    # # # Urban MVLV grids
    #  "1-MVLV-urban-5.303-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-0-no_sw"), 1.5),
    # #"1-MVLV-urban-5.303-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-0-sw"), 1.5),
    # "1-MVLV-urban-5.303-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-1-no_sw"), 1.5),
    # #"1-MVLV-urban-5.303-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-1-sw"), 1.5),
    # "1-MVLV-urban-5.303-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-2-no_sw"), 1.5),
    # #"1-MVLV-urban-5.303-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-5.303-2-sw"), 1.5),
    # "1-MVLV-urban-6.305-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-0-no_sw"), 1.5),
    # #"1-MVLV-urban-6.305-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-0-sw"), 1.5),
    # "1-MVLV-urban-6.305-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-1-no_sw"), 1.5),
    # #"1-MVLV-urban-6.305-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-1-sw"), 1.5),
    # "1-MVLV-urban-6.305-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-2-no_sw"), 1.5),
    # #"1-MVLV-urban-6.305-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.305-2-sw"), 1.5),
    # "1-MVLV-urban-6.309-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-0-no_sw"), 1.5),
    # #"1-MVLV-urban-6.309-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-0-sw"), 1.5),
    # "1-MVLV-urban-6.309-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-1-no_sw"), 1.5),
    # #"1-MVLV-urban-6.309-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-1-sw"), 1.5),
    # "1-MVLV-urban-6.309-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-2-no_sw"), 1.5),
    # #"1-MVLV-urban-6.309-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-6.309-2-sw"), 1.5),
    # #"1-MVLV-urban-all-0-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-0-no_sw"), 1.5),
    # #"1-MVLV-urban-all-0-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-0-sw"), 1.5),
    # #"1-MVLV-urban-all-1-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-1-no_sw"), 1.5),
    # #"1-MVLV-urban-all-1-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-1-sw"), 1.5),
    # #"1-MVLV-urban-all-2-no_sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-2-no_sw"), 1.5),
    # #"1-MVLV-urban-all-2-sw": lambda: increase_line_limits(sb.get_simbench_net("1-MVLV-urban-all-2-sw"), 1.5),

    # local saved grids
    #"caseIEEE37_DG": lambda: use_local_grid("caseIEEE37_DG.m")
}


# Function to increase generation and storage capacities
def increase_generation(net, factor):
    print(f"Verteilte Erzeugung und Speicher um den Faktor {factor} erhöht.")

    # 1. Increase for gen (zentraler Generator)
    for idx, gen in net.gen.iterrows():
        net.gen.at[idx, 'p_mw'] *= factor
        net.gen.at[idx, 'q_mvar'] *= factor
        net.gen.at[idx, 'sn_mva'] = np.sqrt(net.gen.at[idx, 'p_mw'] ** 2 + net.gen.at[idx, 'q_mvar'] ** 2)

    # 2. Increase for sgen (verteilte Erzeugung)
    for idx, sgen in net.sgen.iterrows():
        net.sgen.at[idx, 'p_mw'] *= factor
        net.sgen.at[idx, 'q_mvar'] *= factor
        net.sgen.at[idx, 'sn_mva'] = np.sqrt(net.sgen.at[idx, 'p_mw'] ** 2 + net.sgen.at[idx, 'q_mvar'] ** 2)

    # 3. Increase for storage (Speicher)
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'p_mw'] *= factor
        net.storage.at[idx, 'q_mvar'] *= factor
        net.storage.at[idx, 'sn_mva'] = np.sqrt(net.storage.at[idx, 'p_mw'] ** 2 + net.storage.at[idx, 'q_mvar'] ** 2)

    return net


def increase_line_limits(net, factor):

    print(f"Increase line limits by Factor {factor}.")

    # 🔹 **Reduce Line & Transformer Losses**
    net.line["max_i_ka"] *= 1.5  # increase line limits by 50 %

    if "max_loading_percent" in net.line.columns:
        # Set a default loading percent for all lines (e.g., 100%)
        net.line["max_loading_percent"] *= 1.5

    return net

def use_local_grid(grid_name):
    TESTGRID_PATH = r"C:\Users\runte\Dropbox\Zwischenablage\Testnetze"
    file_path = os.path.join(TESTGRID_PATH, grid_name)
    net = pc.from_mpc(file_path, casename_m="caseIEEE37_DG")

    return net

# Configuration
basic = {
    "Adjustments": True,
    "Overview_Grid": True
}

selected_indicators = {
    "all": False,
    "Self Sufficiency": True,
    "show_self_sufficiency_at_bus": False,
    "System Self Sufficiency": True,
    "Generation Shannon Evenness": True,
    "Generation Variety": True,
    "Line Shannon Evenness": True,
    "Line Variety": True,
    "Load Shannon Evenness": True,
    "Load Variety": True,
    "Disparity Generators": True,
    "Disparity Loads": True,
    "Disparity Transformers": True,
    "Disparity Lines": True,
    "N-3 Redundancy": True,
    "n_3_redundancy_print": False,
    "Redundancy": True,
    "GraphenTheorie": True,
    "Flexibility": True,
    "Flexibility_fxor": True,
    "Buffer": True,
    "show_spider_plot": False,
    "print_results": True,
    "output_excel": True
}

selected_scenario = {
    "stress_scenario": True,
    "all": False,
    "Flood": {"active": True, "runs": 100},
    "Earthquake": {"active": True, "runs": 100},
    "Dunkelflaute": {"active": True, "runs": 10},
    "Storm": {"active": True, "runs": 100},
    "Fire": {"active": False, "runs": 20},
    "Line Overload": {"active": False, "runs": 10},
    "IT-Attack": {"active": False, "runs": 20},
    "Geopolitical_gas": {"active": True, "runs": 10},
    "Geopolitical_h2": {"active": True, "runs": 10},
    "high_EE_generation": {"active": True, "runs": 25},
    "high_load": {"active": True, "runs": 25},
    "sabotage_trafo": {"active": True, "runs": 20},
    "print_results": True,
    "output_excel": True
}


# Main Function
def run_analysis_for_single_grid(grid_name):
    start_timer = time.time()
    dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
    ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

    dfinalresults = add_indicator(dfinalresults, grid_name, 0)

    dfresultsscenario = pd.DataFrame()
    dfresultsscenario = add_indicator(dfresultsscenario, grid_name, 0)

    # Select and create the grid dynamically
    if grid_name in grids:
        net = grids[grid_name]()
    else:
        raise ValueError(f"Unknown Grid Type: {basic['Grid']}")

    if basic["Overview_Grid"]:
        # Count elements and scaled elements
        element_counts = count_elements(net)
        # Print both counts in one row
        print(net)

        print(net.bus)
        print(net.trafo)
        print(net.line)

        # print("Generators:")
        print(net.gen)
        print(net.sgen)
        print(net.storage)


    if basic["Adjustments"]:
        net = set_missing_limits(net)

    if selected_scenario["all"]:
        # Setze alle anderen Indikatoren auf True
        for key, value in selected_scenario.items():
            if isinstance(value, dict):
                value["active"] = True

    if net.bus_geodata.empty:
        selected_scenario["Flood"]["active"] = False
        dfresultsscenario = add_indicator(dfresultsscenario, "Flood", 2)

    if net.trafo.empty:
        selected_scenario["sabotage_trafo"]["active"] = False
        dfresultsscenario = add_indicator(dfresultsscenario, "sabotage_trafo", 2)

    if net.sgen.empty or not net.sgen["type"].str.contains("fuel cell", case=False, na=False).any():
        selected_scenario["Geopolitical_h2"]["active"] = False
        dfresultsscenario = add_indicator(dfresultsscenario, "Geopolitical_h2", 2)

    if net.sgen.empty or not net.sgen["type"].str.contains("CHP|Gasturbine", case=False, na=False).any():
        selected_scenario["Geopolitical_gas"]["active"] = False
        dfresultsscenario = add_indicator(dfresultsscenario, "Geopolitical_gas", 2)

    print(f"{selected_indicators}")

    if selected_indicators["all"]:
        # Setze alle anderen Indikatoren auf True
        for key in selected_indicators:
            if key != "all":  # 'all' selbst bleibt unverändert
                selected_indicators[key] = True

    if selected_indicators["Self Sufficiency"]:
        # Calculate generation factors
        generation_factors = calculate_generation_factors(net, "Fraunhofer ISE (2024)")
        indi_selfsuff = float(selfsuff(net, generation_factors, selected_indicators["show_self_sufficiency_at_bus"]))
        dfinalresults = add_indicator(dfinalresults, 'Self Sufficiency At Bus Level', indi_selfsuff)
        # Prozentzahl, keine Normierung notwendig

    if selected_indicators["System Self Sufficiency"]:
        netsa = net.deepcopy()
        indi_selfsuff_neu = selfsufficiency_neu(netsa)
        dfinalresults = add_indicator(dfinalresults, 'Self Sufficiency System', indi_selfsuff_neu)
        # Prozentzahl, keine Normierung notwendig

    if selected_indicators["Generation Shannon Evenness"] or selected_indicators["Line Shannon Evenness"] or \
            selected_indicators["Load Shannon Evenness"]:
        # Define the maximum known types for each component
        max_known_types = {
            'generation': 10,
            # Adjust this based on your actual known types (sgen: solar, wind, biomass, gen: gas, coal, nuclear, generator, static generator; storage: battery, hydro,
            'line': 2,  # "ol" (overhead line) and "cs" (cable system)
            'load': 10
            # Example: 10 known types of loads (residential, commercial, industrial, agricultaral, transport, municipal, dynamic, static, critical, non-critical
        }

    # Initialize lists to store the values
    evenness_values = []
    variety_values = []

    if selected_indicators["Generation Shannon Evenness"] or selected_indicators["Generation Variety"]:
        generation_data = pd.concat([net.sgen, net.gen, net.storage], ignore_index=True)
        evenness, variety, variety_scaled, max_variety, evenness_entropy = calculate_shannon_evenness_and_variety(generation_data,
                                                                                                max_known_types[
                                                                                                    'generation'])
        evenness_values.append(evenness)
        variety_values.append(variety_scaled)
        dfinalresults = add_indicator(dfinalresults, "Generation Shannon Evenness", evenness_entropy)
        dfinalresults = add_indicator(dfinalresults, "Generation Shannon Evenness scaled", evenness)
        if selected_indicators["Generation Variety"]:
            dfinalresults = add_indicator(dfinalresults, "Generation Variety", variety)
            dfinalresults = add_indicator(dfinalresults, "Generation Variety scaled", variety_scaled)

    if selected_indicators["Line Shannon Evenness"] or selected_indicators["Line Variety"]:
        evenness, variety, variety_scaled, max_variety, evenness_entropy = calculate_shannon_evenness_and_variety(net.line,
                                                                                                max_known_types['line'])
        evenness_values.append(evenness)
        variety_values.append(variety_scaled)
        dfinalresults = add_indicator(dfinalresults, "Line Shannon Evenness", evenness_entropy)
        dfinalresults = add_indicator(dfinalresults, "Line Shannon Evenness scaled", evenness)
        if selected_indicators["Line Variety"]:
            dfinalresults = add_indicator(dfinalresults, "Line Variety", variety)
            dfinalresults = add_indicator(dfinalresults, "Line Variety scaled", variety_scaled)

    if selected_indicators["Load Shannon Evenness"] or selected_indicators["Load Variety"]:
        evenness, variety, variety_scaled, max_variety, evenness_entropy = calculate_shannon_evenness_and_variety(net.load,
                                                                                                max_known_types['load'])
        evenness_values.append(evenness)
        variety_values.append(variety_scaled)
        dfinalresults = add_indicator(dfinalresults, "Load Shannon Evenness", evenness_entropy)
        dfinalresults = add_indicator(dfinalresults, "Load Shannon Evenness scaled", evenness)
        if selected_indicators["Load Variety"]:
            dfinalresults = add_indicator(dfinalresults, "Load Variety", variety)
            dfinalresults = add_indicator(dfinalresults, "Load Variety scaled", variety_scaled)

    if selected_indicators["Generation Shannon Evenness"] or selected_indicators["Generation Variety"] or \
            selected_indicators["Load Shannon Evenness"] or selected_indicators["Load Variety"] or selected_indicators[
        "Line Shannon Evenness"] or selected_indicators["Line Variety"]:
        # Calculate averages if lists are not empty
        if evenness_values:
            avg_evenness = sum(evenness_values) / len(evenness_values)
            dfinalresults = add_indicator(dfinalresults, "Shannon Evenness Average scaled", avg_evenness)

        if variety_values:
            avg_variety = sum(variety_values) / len(variety_values)
            dfinalresults = add_indicator(dfinalresults, "Variety Average scaled", avg_variety)

    if selected_indicators["GraphenTheorie"]:
        # Create an empty NetworkX graph
        G = nx.Graph()

        # 1) Busse als Knoten hinzufügen
        for bus_id in net.bus.index:
            G.add_node(bus_id)

        # 2) Leitungen als Kanten hinzufügen (unter Berücksichtigung geschlossener Schalter)
        for idx, line in net.line.iterrows():
            from_bus = line.from_bus
            to_bus = line.to_bus

            # Prüfen, ob ein Schalter (switch.et == 'l') zwischen den Bussen liegt
            switch_exists = False
            switch_closed = True  # wird nur dann False, wenn wir tatsächlich einen offenen Switch finden

            for _, sw in net.switch.iterrows():
                if sw.et == 'l':
                    # Bus- und Element-Kombination checken
                    if (sw.bus == from_bus and sw.element == to_bus) or (sw.bus == to_bus and sw.element == from_bus):
                        switch_exists = True
                        switch_closed = sw.closed
                        break

            # Nur Kante hinzufügen, wenn kein Switch existiert ODER er geschlossen ist
            if not switch_exists or (switch_exists and switch_closed):
                # Als Gewicht nehmen wir hier exemplarisch die Leitungslänge
                length = line.length_km
                G.add_edge(from_bus, to_bus, weight=length)

        # 3) Trafos als Kanten hinzufügen (ebenfalls optional mit Schalter-Check)
        for idx, trafo in net.trafo.iterrows():
            hv_bus = trafo.hv_bus
            lv_bus = trafo.lv_bus

            # Prüfen, ob ein Schalter (switch.et == 't') zum Trafo existiert
            switch_exists = False
            switch_closed = True

            for _, sw in net.switch.iterrows():
                if sw.et == 't':
                    # Bei Trafos ist meist bus = hv_bus oder lv_bus und element = trafo.id
                    # Hier einfacher Check: falls bus einer der beiden ist und switch.element == diesem Trafo
                    if sw.bus in [hv_bus, lv_bus] and sw.element == idx:
                        switch_exists = True
                        switch_closed = sw.closed
                        break

            # Nur Kante hinzufügen, wenn kein Trafo-Switch existiert ODER dieser geschlossen ist
            if not switch_exists or (switch_exists and switch_closed):
                # Beispiel: Als Gewicht kannst du beliebig etwas hinterlegen (z. B. trafo.sn_mva)
                G.add_edge(hv_bus, lv_bus, weight=1.0)

        # 4) Prüfen, ob der Graph zusammenhängend ist
        if not nx.is_connected(G):
            # Größte zusammenhängende Komponente extrahieren
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component).copy()

        dfinalresults = GraphenTheorieIndicator(G, dfinalresults)

    # Create a list to store individual disparity values
    disparity_values = []

    if selected_indicators["Disparity Generators"]:
        if not selected_indicators["Self Sufficiency"]:
            generation_factors = calculate_generation_factors(net, "Fraunhofer ISE (2024)")

        disparity_df_gen, max_integral_gen = calculate_disparity_space(net, generation_factors)
        integral_value_gen = disparity_df_gen.values.sum()
        ratio_gen = min(1,integral_value_gen / max_integral_gen)
        ddisparity = add_disparity(ddisparity, 'Generators', integral_value_gen, max_integral_gen, ratio_gen)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Generators', integral_value_gen)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Generators scaled', ratio_gen)
        disparity_values.append(ratio_gen)

    if selected_indicators["Disparity Loads"]:
        disparity_df_load, max_integral_load = calculate_load_disparity(net)
        integral_value_load = disparity_df_load.values.sum()
        ratio_load = min(1,integral_value_load / max_integral_load)
        ddisparity = add_disparity(ddisparity, 'Load', integral_value_load, max_integral_load, ratio_load)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Loads', integral_value_load)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Loads scaled', ratio_load)
        disparity_values.append(ratio_load)

    if selected_indicators["Disparity Transformers"]:
        disparity_df_trafo, max_int_trafo = calculate_transformer_disparity(net)
        integral_value_trafo = disparity_df_trafo.values.sum()
        print(disparity_df_trafo)
        print(ddisparity)
        if integral_value_trafo == 0 or disparity_df_trafo.empty:
            print("Disparity Berechnung für Trafos war fehlerhaft und wird mit 0 ersetzt")
            ratio_trafo = 0
            ddisparity = add_disparity(ddisparity, 'Trafo', 0, max_int_trafo, 0)
        else:
            ratio_trafo = min(1,integral_value_trafo / max_int_trafo)
            ddisparity = add_disparity(ddisparity, 'Trafo', integral_value_trafo, max_int_trafo, ratio_trafo)

        dfinalresults = add_indicator(dfinalresults, 'Disparity Transformers', integral_value_trafo)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Transformers scaled', ratio_trafo)
        disparity_values.append(ratio_trafo)

    if selected_indicators["Disparity Lines"]:
        disparity_df_lines, max_int_disp_lines = calculate_line_disparity(net)
        integral_value_line = disparity_df_lines.values.sum()
        ratio_line = min(1,integral_value_line / max_int_disp_lines)
        ddisparity = add_disparity(ddisparity, 'Lines', integral_value_line, max_int_disp_lines, ratio_line)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Lines', integral_value_line)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Lines scaled', ratio_line)
        disparity_values.append(ratio_line)

    # Calculate overall disparity average
    if disparity_values:
        avg_disparity = sum(disparity_values) / len(disparity_values)
        dfinalresults = add_indicator(dfinalresults, 'Disparity Average scaled', avg_disparity)

    if selected_indicators["N-3 Redundancy"]:
        if not basic["Overview_Grid"]:
            # Count elements and scaled elements
            element_counts = count_elements(net)

        # Liste der zu prüfenden Elementtypen
        element_types = ["line", "sgen", "gen", "trafo", "bus", "storage", "switch", "load"]

        n3_redundancy_results = {}
        Success = 0
        Failed = 0
        timeout = 180

        # Über alle relevanten Elementtypen iterieren
        for element_type in element_types:
            start_time = time.time()
            results = n_3_redundancy_check(net, start_time, element_type, timeout, 300)
            n3_redundancy_results[element_type] = results[element_type]

            # Summiere die Ergebnisse
            Success += results[element_type]['Success']
            Failed += results[element_type]['Failed']
            print(time.time() - start_time)

        # Gesamtrate berechnen
        total_checks = Success + Failed
        rate = Success / total_checks if total_checks != 0 else 0

        # Ergebnis in DataFrame speichern
        dfinalresults = add_indicator(dfinalresults, 'Redundancy N-3', rate)
        # Prozentzahl, keine Normierung notwendig

    if selected_indicators["Redundancy"]:
        Lastfluss, n2_Redundanz, kombi, component_indicators, red_results = Redundancy(net, 300)
        dfinalresults = add_indicator(dfinalresults, "Redundancy Loadflow", Lastfluss)
        dfinalresults = add_indicator(dfinalresults, "Redundancy N-2", n2_Redundanz)
        dfinalresults = add_indicator(dfinalresults, "Redundancy Average", kombi)
        # Prozentzahl, keine Normierung notwendig

        # dfinalresults = add_indicator(dfinalresults, "Load Shannon Evenness", evenness)
        # Ausgabe der Indikatoren pro Komponente:
        print("Komponentenindikatoren (1 = optimal, 0 = schlecht):")
        for comp, inds in component_indicators.items():
            print(f"{comp.capitalize()}:")
            print(f"  Lastfluss: {inds['lf']:.3f}")
            print(f"  Redundanz: {inds['red']:.3f}")
            print(f"  Kombiniert: {inds['combined']:.3f}")

        # Ergebnisse ausgeben
        print("\nErgebnisse der N-2-Redundanzprüfung:")
        for element, stats in red_results.items():
            print(f"{element.capitalize()}: Erfolg: {stats['Success']}, Fehlgeschlagen: {stats['Failed']}")

        print("\nGesamtindikatoren:")
        print(f"  Lastfluss Gesamt: {Lastfluss:.3f}")
        print(f"  N-2 Redundanz Gesamt: {n2_Redundanz:.3f}")
        print(f"  Kombinierter Gesamtindikator: {kombi:.3f}")

    if selected_indicators["Flexibility"]:
        dflexiresults = calculate_flexibility(net)
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Grid Reserves', dflexiresults.loc[
            dflexiresults['Indicator'] == 'Flex Netzreserve', 'Value'].values[0])
        # Prozentzahl, keine Normierung notwendig
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Reserve Critical Lines scaled', dflexiresults.loc[
            dflexiresults['Indicator'] == 'Flex Reserve krit Leitungen scaled', 'Value'].values[0])
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Reserve Critical Lines', dflexiresults.loc[
            dflexiresults['Indicator'] == 'Flex Reserve krit Leitungen', 'Value'].values[0])
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Average', dflexiresults.loc[
            dflexiresults['Indicator'] == 'Flexibilität Gesamt', 'Value'].values[0])

    if selected_indicators["Buffer"]:
        speicher_scaled, speicher = calculate_buffer(net)
        dfinalresults = add_indicator(dfinalresults, 'Buffer Capacity scaled', speicher_scaled)
        dfinalresults = add_indicator(dfinalresults, 'Buffer Capacity', speicher)

    if selected_indicators["Flexibility_fxor"]:
        Flex_fxor_scaled, Flex_fxor = flexibility_fxor(net, False)
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Feasible Operating Region scaled', Flex_fxor_scaled)
        dfinalresults = add_indicator(dfinalresults, 'Flexibility Feasible Operating Region', Flex_fxor)

    if selected_indicators["n_3_redundancy_print"]:
        print("Results of N-3 Redundancy")
        for element_type, counts in n3_redundancy_results.items():
            print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")
    if selected_indicators["show_spider_plot"]:
        plot_spider_chart(dfinalresults)

    if not dfinalresults.empty:
        runtime_required = (time.time() - start_timer)
        dfinalresults = add_indicator(dfinalresults, "Time required", runtime_required)

        # Separate the first and last row
        first_row = dfinalresults.iloc[[0]]

        # Sort everything in between
        middle_rows = dfinalresults.iloc[1:].sort_values(by="Indicator").reset_index(drop=True)

        # Recombine everything
        dfinalresults = pd.concat([first_row, middle_rows], ignore_index=True)

    if selected_indicators["print_results"]:
        print("Results for Indicators:")
        print(dfinalresults)


    
    if selected_scenario["stress_scenario"]:

        for scenario, params in selected_scenario.items():
            if isinstance(params, dict) and params.get("active", False):
                stressor = scenario.lower()
                scenario_values = []
                print(f"{stressor}")

                for n in range(params.get("runs", 10)):  # fallback to 10 runs if "runs" not defined
                    modified_nets = stress_scenarios(net, [stressor])
                    # `modified_nets` is a list of (scenario_name, modified_net) tuples.

                    if not modified_nets:
                        print("No modified net returned. Skipping this scenario.")
                        continue

                    # Extract the first (and presumably only) tuple
                    scenario_name, single_net = modified_nets[0]

                    # Now you can run the OPF using `single_net`
                    res_scenario = run_scenario(single_net, scenario_name)
                    scenario_values.append(res_scenario)
                    del modified_nets  # optional

                # Compute the average for this scenario
                print(f"{scenario_values}")
                avg_value = sum(scenario_values) / len(scenario_values)
                dfresultsscenario = add_indicator(dfresultsscenario, scenario, avg_value)

        if not dfresultsscenario.empty:
            # Separate the first and last row
            print(dfresultsscenario)
            first_row = dfresultsscenario.iloc[[0]]

            # Sort everything in between
            middle_rows = dfresultsscenario.iloc[1:].sort_values(by="Indicator").reset_index(drop=True)

            # Recombine everything
            dfresultsscenario = pd.concat([first_row, middle_rows], ignore_index=True)

            # Compute the average of all values excluding the first row
            if len(dfresultsscenario) > 1:  # Ensure there are enough rows to calculate an average
                valid_values = dfresultsscenario["Value"].iloc[1:] # Exclude the first row
                filtered_values = valid_values[(valid_values >= 0) & (valid_values <= 1)] # Exclude values above 1 = not calculated scenarios
                scenario_average_value = filtered_values.mean()

            # Add the average as a new row
            dfresultsscenario = add_indicator(dfresultsscenario, "Overall Scenario Resilience Score",
                                              scenario_average_value)

            dfresultsscenario["Value"] = dfresultsscenario["Value"].replace(2, np.nan)

            if selected_scenario["print_results"]:
                print(dfresultsscenario)

    if selected_scenario["output_excel"] or selected_indicators.get("output_excel"):
        # Output-Dateiname basierend auf grid_name
        output_filename = f'Ergebnisse_{grid_name}.xlsx'
        output_dir = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots"
        output_path = os.path.join(output_dir, output_filename)

        # ExcelWriter verwenden, um mehrere Sheets in eine Datei zu schreiben
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if selected_indicators.get("output_excel"):
                dfresultsscenario.T.to_excel(writer, sheet_name="Results Scenario", index=False)

            if selected_indicators["output_excel"]:
                dfinalresults.T.to_excel(writer, sheet_name="Results Indicator", index=False)



def run_all_grids(start_time):
    """
    Loops over the grids dictionary and runs the above 'run_analysis_for_single_grid' on each.
    """
    for grid_name in grids:
        timer =  time.time() - start_time
        print(f"\n--- Running analysis for grid: {grid_name} at {timer}---")

        run_analysis_for_single_grid(grid_name)


def main():
    """
    The 'entry point' that is invoked when you run this script.
    """
    # Optionally, you can decide whether to process all grids or just one,
    # e.g. based on some config or command line argument
    start_time = time.time()
    process_all = True  # or read from config/CLI

    if process_all:
        run_all_grids(start_time)
        # do final post-processing, exporting, etc.
    else:
        # Suppose your config says to just run the 'case30' grid
        grid_name = "caseIEEE37_DG"
        run_analysis_for_single_grid(grid_name)


if __name__ == '__main__':
    main()
