
from datasets import load_dataset

dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

print(dataset)
print(dataset["test"][0])
print(dataset["test"].features)