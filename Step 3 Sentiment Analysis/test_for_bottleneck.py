from concurrent.futures import ThreadPoolExecutor
import time
import ollama

def test_ollama_concurrency(prompt: str, n_threads: int = 4, n_requests: int = 8):
    """Test if ollama benefits from concurrent requests."""
    def make_request():
        start = time.time()
        response = ollama.generate(
            model="qwen2.5:3b",
            prompt=prompt,
            options={'temperature': 0.2}
        )
        return time.time() - start

    # Sequential requests
    print("Testing sequential requests...")
    t0 = time.time()
    sequential_times = []
    for _ in range(n_requests):
        sequential_times.append(make_request())
    sequential_total = time.time() - t0
    
    # Parallel requests
    print("\nTesting parallel requests...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        parallel_times = list(executor.map(lambda _: make_request(), range(n_requests)))
    parallel_total = time.time() - t0
    
    print(f"\nResults for {n_requests} requests:")
    print(f"Sequential total time: {sequential_total:.2f}s (avg {sum(sequential_times)/len(sequential_times):.2f}s per request)")
    print(f"Parallel total time: {parallel_total:.2f}s (avg {sum(parallel_times)/len(parallel_times):.2f}s per request)")
    print(f"Speedup: {sequential_total/parallel_total:.2f}x")

# Example usage:
test_prompt = "Rate this sentence from 1-5: This is a test prompt"
test_ollama_concurrency(test_prompt, n_threads=8, n_requests=100)