# scripts/extract_insert_lines.py
input_file = './src/data/game_events_archive.sql'
output_file = './src/data/game_events_insert.sql'

with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if line.strip().startswith('INSERT INTO'):
            outfile.write(line)
