def solve(s):
    """
    Determine whether a situation is dangerous based on player positions.
    
    A situation is dangerous if there are at least 7 players of the same team
    standing one after another.
    
    Preconditions:
        - The input string is non-empty.
        - The input string consists only of characters '0' and '1'.
        - The length of the input string does not exceed 100 characters.
        - At least one player from each team is present on the field.
    
    Invariants:
        - A situation is dangerous if and only if there are at least 7 players
          of some team standing one after another.
    
    Postconditions:
        - Returns 'YES' if the situation is dangerous.
        - Returns 'NO' if the situation is not dangerous.
    
    Resource Bounds:
        - Execution time must not exceed 2 seconds.
        - Memory usage must not exceed 256 megabytes.
    """
    if "0000000" in s or "1111111" in s:
        return "YES"
    return "NO"


def main():
    s = input().strip()
    result = solve(s)
    print(result)


if __name__ == "__main__":
    main()
