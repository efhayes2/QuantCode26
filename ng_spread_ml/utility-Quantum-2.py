def get_month_map():
    return  {
        'F': '01', 'G': '02', 'H': '03', 'J': '04', 'K': '05', 'M': '06',
        'N': '07', 'Q': '08', 'U': '09', 'V': '10', 'X': '11', 'Z': '12'
    }

def get_march_position_map():
    return {
        'F': 3, 'G': 2, 'H': 1, 'J': 12, 'K': 11, 'M': 10,
        'N': 9, 'Q': 8, 'U': 7, 'V': 6, 'X': 5, 'Z': 4
    }


def get_relative_contract_map(start_month_code):
    # Standard futures month codes (Jan-Dec)
    codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

    if start_month_code not in codes:
        raise ValueError(f"Invalid month code: {start_month_code}")

    start_idx = codes.index(start_month_code)
    relative_map = {}

    for code in codes:
        # Distance from the current code forward to the start_month_code
        # Using (start_idx - current_idx) matches your logic:
        # If start is 'F' (0) and current is 'Z' (11), distance is 1 (wrapping)
        position = (start_idx - codes.index(code)) % 12 + 1
        relative_map[code] = position

    return relative_map
