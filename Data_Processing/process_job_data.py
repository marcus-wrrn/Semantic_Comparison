import pandas as pd
import json

def get_article(word):
    """Determine whether to use 'a' or 'an' before the given word."""
    vowels = ['a', 'e', 'i', 'o', 'u']
    first_letter = word[0].lower()
    if first_letter in vowels:
        return 'an'
    else:
        return 'a'



# Get jobs
df = pd.read_csv("/home/marcuswrrn/Projects/Semantic_Quantification/Semantic_Comparison/Data/Job_Titles.csv", index_col=None)
jobs = df["Job_Titles"].to_list()



# Generate Sentences

prompts = [{"role": "user", "content": f"I am {get_article(job)} {job}"} for job in jobs]

with open("/home/marcuswrrn/Projects/Semantic_Quantification/Semantic_Comparison/Data/jobs_prompts.json", "w") as file:
    json.dump(prompts, file)

