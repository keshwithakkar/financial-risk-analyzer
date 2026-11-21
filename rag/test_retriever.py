
from retriever import retrieve_policy

query = "How does previous loan default affect credit risk?"

results = retrieve_policy(query)

print("\nRelevant Financial Policy Sections:\n")

for result in results:
    print(result)
    print("-------------------------")
