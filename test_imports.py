import os
import sys

# Print current working directory
print("Current working directory:", os.getcwd())

# Print Python path
print("\nPython path:")
for path in sys.path:
    print(path)

# Try importing from stock_server
print("\nTrying to import stock_server:")
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "FinDataMCP"))
    from stock_server import get_stock_info
    print("Successfully imported get_stock_info")
    
    # Test the function
    print("\nTesting get_stock_info with AAPL:")
    result = get_stock_info("AAPL")
    print(result)
except Exception as e:
    print("Error:", str(e))
