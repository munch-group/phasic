import re
import sys

_, major, minor, fix = sys.argv
major, minor, fix = int(major), int(minor), int(fix)
assert sum([major, minor, fix]) == 1

# file, regex pairs
spec = {
    'src/python/ptdalgorithms/__init__.py':  r"(__version__\s*=\s*')(\d+)\.(\d+)\.(\d+)(')",
    'DESCRIPTION': r"(Version: )(\d+)\.(\d+)\.(\d+)(.*)",
    'CMakeLists.txt': r"(ptdalgorithms\s+VERSION\s+)(\d+)\.(\d+)\.(\d+)(.*)"
}

def bump(content, m):
    assert m is not None, "Version not found"
    prefix = m.group(1)
    _major = int(m.group(2))
    _minor = int(m.group(3))
    _fix = int(m.group(4))
    postfix = m.group(5)
    version = f'{_major}.{_minor}.{_fix}'
    new_version = f'{_major+major}.{_minor+minor}.{_fix+fix}'
    match = f'{prefix}{version}{postfix}'
    repl = f'{prefix}{new_version}{postfix}'
    new_content = content.replace(match, repl)
    return new_content, new_version

new_contents = {}
new_versions = set()
for file, regex in spec.items():
    with open(file, 'r') as f:
        content = f.read()
    m = re.search(regex, content)
    new_content, new_version = bump(content, m)
    new_contents[file] = new_content
    new_versions.add(new_version)

# all versions should be the same
assert len(new_versions) == 1
new_version = list(new_versions)[0]

for file, content in new_contents.items():
    with open(file, 'w') as f:
        f.write(content)

_major, _minor, _fix = map(int, new_version.split('.'))
old_version = f'{_major-major}.{_minor-minor}.{_fix-fix}'

print(f"Version bump:\n  {old_version} -> {new_version}:\nFiles changed:")
for file, content in new_contents.items():
    print(f"  {file}")
