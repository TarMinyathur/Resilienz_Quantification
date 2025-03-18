import pandapower as pp

def run_scenario(net, scenario_name):
    """
    Runs an optimal power flow (OPF) on 'net' using two attempts:
      1) init='pf'
      2) init='flat' if the first attempt fails

    Returns:
        1 for success (OPF converged)
        0 for failure (OPF failed to converge or raised an exception)
    """
    try:
        # First attempt with init='pf'
        pp.runopp(
            net,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        print(f"Scenario '{scenario_name}': OPF => SUCCESS (init='pf')")
        return 1

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        # If first attempt fails, try init='flat'
        try:
            pp.runopp(
                net,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            print(f"Scenario '{scenario_name}': OPF => SUCCESS (init='flat')")
            return 1
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            print(f"Scenario '{scenario_name}': OPF => FAILED (both init='pf' and init='flat')")
            return 0
        except Exception as e:
            print(f"Scenario '{scenario_name}': Unexpected error in second attempt: {e}")
            return 0

    except Exception as e:
        print(f"Scenario '{scenario_name}': Unexpected error in first attempt: {e}")
        return 0
