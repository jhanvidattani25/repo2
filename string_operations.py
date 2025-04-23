def string_operations(string):
    print(f"String: string.strip()") 
    print(f"Lower case: {string.lower()}")
    print(f"Upper case: {string.upper()}")
    print(f"Length: {len(string)}")
    print(f"Reversed: {string[::-1]}")
    print(f"Title case: {string.title()}")
    print(f"Is alphanumeric: {string.isalnum()}")
    print(f"Is digit: {string.isdigit()}")
    print(f"Is alpha: {string.isalpha()}")

if __name__ == "__main__":
    string = "test"
    string_operations(string)
