def add(x, y):
    """Adds two numbers."""
    return x + y

def subtract(x, y):
    """Subtracts two numbers."""
    return x - y

def multiply(x, y):
    """Multiplies two numbers."""
    return x * y

def divide(x, y):
    """Divides two numbers. Handles division by zero."""
    if y == 0:
        return "Error! Division by zero."
    else:
        return x / y


def calculate_power(base, exponent):
    """
    Calculates the power of a number (base raised to the exponent).

    Args:
        base (int or float): The base number.
        exponent (int): The exponent (must be an integer).

    Returns:
        int or float: The result of base raised to the power of exponent.
                      Returns None if the exponent is not an integer.
    """
    if not isinstance(exponent, int):
        print("Error: The exponent must be an integer.")
        return None
    
    return base ** exponent


def calculator():
    """
    A simple command-line calculator.
    Allows the user to choose an operation and input two numbers.
    """
    print("Welcome to the Python Calculator!")
    print("Select operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Exit")

    while True:
        choice = input("Enter choice(1/2/3/4/5): ")

        if choice in ('1', '2', '3', '4'):
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
            except ValueError:
                print("Invalid input. Please enter numbers only.")
                continue

            if choice == '1':
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == '2':
                print(f"{num1} - {num2} = {subtract(num1, num2)}")
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == '4':
                result = divide(num1, num2)
                if isinstance(result, str):  # Check if it's the error message
                    print(result)
                else:
                    print(f"{num1} / {num2} = {result}")
        elif choice == '5':
            print("Exiting calculator. Goodbye!")
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    calculator()
