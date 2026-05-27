"""Utility function to convert strings to a safe format for filenames and identifiers."""
import re

def safe_name(x):
    """Convert a string to a safe format for filenames and identifiers."""
    x = str(x)
    x = x.strip().replace(" ", "_")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^A-Za-z0-9_.-]", "_", x)
    x = re.sub(r"_+", "_", x)
    return x
