"""Grid cell constants and grid state parsing utilities."""

import re

# Cell identity symbol to ID mapping
CELL_SYMBOL_TO_ID = {
    "A": 0,  # Agent
    "#": 1,  # Wall
    "G": 2,  # Goal
    "_": 3,  # Empty
    "D": 4,  # Door
    "K": 5,  # Key
    "?": 6,  # Unknown
    "+": 7,  # Padding
}

# Reverse mapping for convenience
CELL_ID_TO_SYMBOL = {v: k for k, v in CELL_SYMBOL_TO_ID.items()}


def parse_grid_state(
    grid_state: list[str],
    pad_to_size: int | None = None,
) -> list[list[int]]:
    """Parse grid_state strings into [row_id, column_id, cell_identity_id] triples.

    Handles both single-digit and double-digit row/column indices in the grid format.

    Args:
        grid_state: List of strings representing the grid, where the first line
            is the column header and subsequent lines are rows with format
            "row_id cell cell cell ..."
        pad_to_size: If specified, pad the grid to this size using the padding
            symbol '+' (id 7). Padding is added to the right and bottom.

    Returns:
        List of [row_id, column_id, cell_identity_id] triples, ordered row by row,
        column by column.

    Example:
        >>> grid = [
        ...     "  0 1 2 3 4 ",
        ...     "0 # # # # # ",
        ...     "1 # _ _ G # ",
        ...     "2 # _ # # # ",
        ...     "3 # _ A _ # ",
        ...     "4 # # # # # "
        ... ]
        >>> triples = parse_grid_state(grid)
        >>> triples[0]  # First cell (0, 0) is a wall
        [0, 0, 1]
    """
    if not grid_state:
        raise ValueError("grid_state is empty")

    # Skip the header line (column indices)
    data_lines = grid_state[1:]

    # Parse each row
    parsed_cells: list[list[int]] = []
    actual_grid_size = 0

    for line_raw in data_lines:
        line = line_raw.rstrip()
        if not line:
            continue

        # Extract row index and cells
        # The row starts with the row number, followed by cells
        # Use regex to find the row number at the start
        match = re.match(r"^(\d+)\s+", line)
        if not match:
            continue

        row_id = int(match.group(1))
        actual_grid_size = max(actual_grid_size, row_id + 1)

        # Get the rest of the line after the row number
        rest = line[match.end() :]

        # Extract cell symbols - they are single characters separated by spaces
        # Split by whitespace and filter to get only valid cell symbols
        tokens = rest.split()
        col_id = 0
        for token in tokens:
            # Each token should be a single character cell symbol
            if len(token) == 1 and token in CELL_SYMBOL_TO_ID:
                cell_id = CELL_SYMBOL_TO_ID[token]
                parsed_cells.append([row_id, col_id, cell_id])
                col_id += 1
                actual_grid_size = max(actual_grid_size, col_id)

    # Determine the target size
    target_size = pad_to_size if pad_to_size is not None else actual_grid_size

    if target_size < actual_grid_size:
        raise ValueError(f"pad_to_size ({target_size}) is smaller than actual grid size ({actual_grid_size})")

    # Build a complete grid with padding
    result: list[list[int]] = []
    padding_id = CELL_SYMBOL_TO_ID["+"]

    # Create a lookup for existing cells
    cell_lookup = {(cell[0], cell[1]): cell[2] for cell in parsed_cells}

    for row_id in range(target_size):
        for col_id in range(target_size):
            if (row_id, col_id) in cell_lookup:
                cell_id = cell_lookup[(row_id, col_id)]
            else:
                cell_id = padding_id
            result.append([row_id, col_id, cell_id])

    return result
