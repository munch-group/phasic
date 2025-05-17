#!/usr/bin/env python

from subprocess import check_output, CalledProcessError

def run_command(cmd):
    try:
        output = check_output(cmd, shell=True).decode()
        return output.strip()    
    except CalledProcessError as e: 
        exit(1)

run_command('git status --porcelain')

with open('DESCRIPTION', 'r') as f:
    for line in f.readlines():
        if line.startswith('Version:'):
            version = line.split()[1]
            tag = f"v{version}"
            break

run_command(f'git tag -a {tag} -m "${{1:-Release}}"')
run_command(f'git push origin --tags')
