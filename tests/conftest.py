import pytest
import time

@pytest.fixture(autouse=True)
def log_test_name_and_time(request):
    test_name = request.node.name
    print(f"Starting test: {test_name}")
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"Finished test: {test_name} in {elapsed_time:.2f} seconds")
