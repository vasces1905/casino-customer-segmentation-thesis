# clean_tex_file.py
input_path = 'Dissertation.tex'
output_path = 'Dissertation_cleaned.tex'

with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    cleaned_line = ''.join(c for c in line if ord(c) < 128)  # remove non-ASCII chars
    cleaned_lines.append(cleaned_line)

with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print("Cleaned file saved as Dissertation_cleaned.tex")
