# count_elements.py

import math

def count_elements(net):
    counts = {
        "switch": len(net.switch) if "switch" in net else 0,  # Handle networks without storage elements
        "External_grid": len(net.ext_grid) if "ext_grid" in net else 0,  # Handle networks without storage elements
        "load": len(net.load) if "load" in net else 0,  # Handle networks without storage elements
        "sgen": len(net.sgen) if "sgen" in net else 0,  # Handle networks without storage elements
        "gen": len(net.gen) if "gen" in net else 0,  # Handle networks without storage elements
        "line": len(net.line) if "line" in net else 0,  # Handle networks without storage elements
        "trafo": len(net.trafo) if "trafo" in net else 0,  # Handle networks without storage elements
        "bus": len(net.bus) if "bus" in net else 0,  # Handle networks without storage elements
        "storage": len(net.storage) if "storage" in net else 0  # Handle networks without storage elements
    }

    # # Multiply counts by 0.3 and round down
    # scaled_counts = {}
    # for element_type, count in counts.items():
    #     scaled_count = math.floor(count * 0.15)
    #     scaled_counts[element_type] = scaled_count

    # Create a dictionary containing both counts and scaled counts
    element_counts = {
        "original_counts": counts,
    }

    return element_counts