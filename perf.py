import time

# Benchmark Inference Speed
start_time = time.time()
for _ in range(100):
    usm.query("machine learning models")
latency = (time.time() - start_time)/100

# Accuracy Test
test_cases = [
    ("decision making", "What do AI systems use?"),
    ("structured information", "Knowledge graphs represent...")
]

correct = 0
for answer, question in test_cases:
    results = usm.query(question)
    correct += int(answer in results)

print(f"\nPerformance Metrics:")
print(f"Latency: {latency*1000:.2f}ms per query")
print(f"Accuracy: {correct/len(test_cases)*100}%")