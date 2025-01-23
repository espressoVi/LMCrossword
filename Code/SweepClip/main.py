from utils.dataset import Dataset 
from utils.model import GPT, LLaMA


def main():
    llm = GPT()
    dataset = Dataset()
    for i in range(100):
        print("Currently computing :",i)
        print(dataset[i].number)
        dataset[i].solve(llm)

if __name__ == "__main__":
    main()
