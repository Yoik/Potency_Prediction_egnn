import pandas as pd

df = pd.read_csv("data/raw_assays.csv")
df.columns = [c.strip() for c in df.columns]

# ⭐ 核心修复
if 'Compound' not in df.columns and 'Unnamed: 0' in df.columns:
    df = df.rename(columns={'Unnamed: 0': 'Compound'})

# 只用 GRK2
df['Efficacy'] = df['miniGo12'] / 100.0

name_map = {
    "Dopamine": "Dopa", "Aripiprazole": "ARI", "Brexpiprazole": "BRE",
    "Cariprazine": "CAR", "Lisuride": "Lisu", "Rotigotine": "ROT",
    "UNC2458A": "UNC", "(S)-IHCH-7084": "S84","Pramipexole": "PPX",
    "(R)-IHCH-7010": "R10", "(S)-IHCH-7010": "S10"
}

df['Compound'] = df['Compound'].apply(lambda x: name_map.get(x.strip(), x.strip()))

df[['Compound', 'Efficacy']].to_csv("data/labels.csv", index=False)

print(df[['Compound', 'Efficacy']])
