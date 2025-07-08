import pandas as pd

input_file = "src/data/customer_demographics_2022.csv"
output_file = "src/data/customer_demographics_2022_clean.csv"

df = pd.read_csv(input_file)

def fix_id(cid):
    try:
        number = int(str(cid).replace("CUST_", "").strip())
        return f"CUST_{str(number).zfill(6)}"
    except:
        return "CUST_000000"

df['customer_id'] = df['customer_id'].apply(fix_id)
df.to_csv(output_file, index=False)
print(f"✅ Cleaned file saved to {output_file}")

