from collections import defaultdict

def mapper(word):
    return (word, 1)

words = ["the", "cat", "hissed", "at", "the", "dog"]

mapped = list(map(mapper, words))
print(mapped)

def group(mapped_data):
    grouped = defaultdict(list)
    for key, value in mapped_data:
        grouped[key].append(value)
    return grouped

grouped = group(mapped)
print(grouped)

def reduce(grouped_data):
    return {key: sum(values) for key, values in grouped_data.items()}

reduced = reduce(grouped)
print(reduced)
