import random
import re

def create_random_fen():
    """Generates a random FEN string.
    Each of the 13 states (pieces or empty) has equal probability.
    """
    # Define all pieces and '1' for empty.
    choices = "PNBRQKpnbrqk1"
    
    # Generate 8 rows of 8 random characters.
    rows = ["".join(random.choices(choices, k=8)) for _ in range(8)]
    
    # Join rows with '/' and compress consecutive '1's using regex.
    raw_board = "/".join(rows)
    board = re.sub(r"1+", lambda m: str(len(m.group(0))), raw_board)
    
    # Return FEN with standard defaults for non-placement fields.
    return f"{board}"



if __name__ == "__main__":    
    num_fens = 30  # Number of random FENs to generate
    output_file = "random_fens.csv"
    #Creates a CSV file with random FENs and matching headers.
    with open(output_file, "w") as f:
        f.write("from_frame,to_frame,fen\n")
        for i in range(num_fens):
            f.write(f"{i},{i},{create_random_fen()}\n")
