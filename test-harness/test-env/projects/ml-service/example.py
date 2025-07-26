"""
Example Python module for ml-service

This file demonstrates indexable code for testing the claude-code-context system.
"""

class DataProcessor:
    """Process and analyze data with various algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process_data(self, data: list) -> dict:
        """
        Process input data and return summary statistics.
        
        Args:
            data: List of numeric values to process
            
        Returns:
            Dictionary containing processing results
        """
        if not data:
            return {"error": "No data provided"}
        
        result = {
            "count": len(data),
            "sum": sum(data),
            "average": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }
        
        self.processed_count += 1
        return result
    
    async def async_process(self, data: list) -> dict:
        """Asynchronous version of data processing."""
        import asyncio
        await asyncio.sleep(0.1)  # Simulate async work
        return self.process_data(data)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
if __name__ == "__main__":
    processor = DataProcessor("example")
    test_data = [1, 2, 3, 4, 5]
    result = processor.process_data(test_data)
    print(f"Processed data: {result}")
