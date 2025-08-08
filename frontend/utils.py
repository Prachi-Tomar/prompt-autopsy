import os

def get_backend_host():
    return os.getenv("BACKEND_HOST", "http://localhost:8000")


def make_ascii_table(data, headers=None):
    """
    Create an ASCII table from a 2D list of data.
    
    Args:
        data: 2D list of data rows
        headers: Optional list of column headers
    
    Returns:
        String representation of the ASCII table
    """
    if not data:
        return ""
    
    # Combine headers with data if provided
    if headers:
        all_data = [headers] + data
    else:
        all_data = data
    
    # Calculate column widths
    col_widths = []
    for col_idx in range(len(all_data[0])):
        max_width = max(len(str(row[col_idx])) if col_idx < len(row) else 0 for row in all_data)
        col_widths.append(max_width)
    
    # Create format string
    format_str = "| " + " | ".join(f"{{:<{width}}}" for width in col_widths) + " |"
    
    # Create separator line
    separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
    
    # Build table
    lines = []
    if headers:
        lines.append(format_str.format(*headers))
        lines.append(separator)
        data_rows = all_data[1:]
    else:
        data_rows = all_data
    
    for row in data_rows:
        # Pad row with empty strings if needed
        padded_row = row + [""] * (len(col_widths) - len(row))
        lines.append(format_str.format(*padded_row))
    
    return "\n".join(lines)