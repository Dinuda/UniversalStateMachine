# Initialize USM
usm = UniversalStateMachine()

# Train with sample data
corpus = [
    "AI systems use algorithms for decision making.",
    "Transformers rely on attention mechanisms.",
    "Knowledge graphs represent structured information."
]

for text in corpus:
    usm.add_data(text)

# Test Query
test_query = "What do AI systems use?"
results = usm.query(test_query)
print(f"Query Response: {results}")