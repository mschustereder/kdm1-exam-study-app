#!/usr/bin/env python3
import json
from pathlib import Path

CONCISE_DIR = Path('concise_summaries')
TOPICS_DIR = Path('topics')
TOPICS_DIR.mkdir(exist_ok=True)

HEADINGS = {
    '01_Introduction': 'Introduction',
    '02_Dataset-Collection': 'Dataset Collection',
    '03_Visual-Preprocessing': 'Visual Preprocessing',
    '04_Statistical-Data-Science': 'Statistical Data Science',
    '05_Outliers_Missing-Values': 'Outliers & Missing Values',
    '06_Feature_Engineering': 'Feature Engineering',
    '07_Dim_Reduction': 'Dimensionality Reduction',
    '08_Clustering_Classification': 'Clustering & Classification',
    '09_Pattern_Mining': 'Pattern Mining',
    '10_Evaluation': 'Evaluation',
}

ICONS = {
    '01_Introduction': '\U0001f4da',
    '02_Dataset-Collection': '\U0001f4ca',
    '03_Visual-Preprocessing': '\U0001f441\ufe0f',
    '04_Statistical-Data-Science': '\U0001f4c8',
    '05_Outliers_Missing-Values': '\u26a0\ufe0f',
    '06_Feature_Engineering': '\U0001f527',
    '07_Dim_Reduction': '\U0001f4c9',
    '08_Clustering_Classification': '\U0001f3af',
    '09_Pattern_Mining': '\U0001f50d',
    '10_Evaluation': '\u2705',
}

for md_file in sorted(CONCISE_DIR.glob('*.md')):
    stem = md_file.stem

    with open(md_file, 'r') as f:
        text = f.read()

    lines = text.split('\n')
    sections = []
    current_heading = None
    current_content = []

    for line in lines:
        if line.startswith('## '):
            if current_heading is not None:
                sections.append({
                    'heading': current_heading,
                    'content': '\n'.join(current_content).strip()
                })
            current_heading = line[3:].strip()
            current_content = []
        elif line.startswith('# '):
            continue
        else:
            current_content.append(line)

    if current_heading is not None:
        sections.append({
            'heading': current_heading,
            'content': '\n'.join(current_content).strip()
        })

    num = int(stem[:2])

    topic = {
        'id': num,
        'title': HEADINGS[stem],
        'icon': ICONS[stem],
        'depths': {
            'concise': {
                'available': True,
                'sections': sections
            },
            'medium': {
                'available': False,
                'sections': []
            },
            'full': {
                'available': False,
                'sections': []
            }
        }
    }

    out_file = TOPICS_DIR / f'topic_{num}.json'
    with open(out_file, 'w') as f:
        json.dump(topic, f, indent=2, ensure_ascii=False)

    print(f'Created {out_file} ({len(sections)} sections)')

print('\nDone! All 10 topic JSON files created.')
