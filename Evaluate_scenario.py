import pandapower as pp
import concurrent.futures


def run_scenario(modified_grids, scenario):
    results = {scenario: {"Success": 0, "Failed": 0}}


    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Unpack name and grid correctly from modified_grids
        futures = {executor.submit(process_scenario, grid): name for name, grid in modified_grids}

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            _, status = future.result()
            results[scenario][status] += 1

    # Calculate convergence percentage
    total_cases = results[scenario]["Success"] + results[scenario]["Failed"]
    convergence_rate = (results[scenario]["Success"] / total_cases) * 100 if total_cases > 0 else 0

    return convergence_rate


def process_scenario(net_temp_red):
    """ Runs an optimal power flow (OPF) simulation and returns success/failure status. """
    try:
        # First attempt with init="pf"
        pp.runopp(
            net_temp_red,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        return "Scenario", "Success"

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        try:
            # Retry with init="flat"
            pp.runopp(
                net_temp_red,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            return "Scenario", "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            return "Scenario", "Failed"
        except Exception:
            return "Scenario", "Failed"
    except Exception:
        return "Scenario", "Failed"
