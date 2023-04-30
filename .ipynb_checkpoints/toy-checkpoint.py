def greet(name):
    print(f"Hello, {name}!")

def calculate_sum(a, b):
    return a + b

def concatenate(a,b):
    return str(a) + str(b)
    
if __name__ == '__main__':
    print("inside manin")
    name = concatenate("DK","KH")
    greet(name)
    greet("Bob")
    result = calculate_sum(3, 5)
    print(f"The sum is {result}.")