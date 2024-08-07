# count_elements.py

import math

def count_elements(net):
    counts = {
        "switch": len(net.switch),
        "load": len(net.load),
        "sgen": len(net.sgen),
        "line": len(net.line),
        "trafo": len(net.trafo),
        "bus": len(net.bus),
        "storage": len(net.storage) if "storage" in net else 0  # Handle networks without storage elements
    }

    # Multiply counts by 0.3 and round down
    scaled_counts = {}
    for element_type, count in counts.items():
        scaled_count = math.floor(count * 0.3)
        scaled_counts[element_type] = scaled_count

    # Create a dictionary containing both counts and scaled counts
    element_counts = {
        "original_counts": counts,
        "scaled_counts": scaled_counts
    }

    return element_counts