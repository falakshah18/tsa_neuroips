import re, os

with open('repomix-output-fixed.xml', 'r', encoding='utf-8') as f:
    content = f.read()

matches = re.findall(r'<file path="([^"]+)">\n(.*?)\n</file>', content, re.DOTALL)
print(f'Found {len(matches)} files')

for path, file_content in matches:
    full_path = path.replace('/', os.sep)
    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else '.', exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    print(f'  wrote {full_path}')

print('Done!')