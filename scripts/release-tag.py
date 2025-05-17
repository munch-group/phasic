
##!/usr/bin/env python

from subprocess import check_output, CalledProcessError
import toml


def run_command(cmd):
    try:
        output = check_output(cmd, shell=True).decode()
        return output.strip()    
    except CalledProcessError as e: 
        exit(1)

run_command('git status --porcelain')

with open('pyproject.toml', 'r') as f:
    pyproject = toml.load(f)

tag = f"v{pyproject['project']['version']}"

run_command(f'git tag -a {tag} -m "${{1:-Release}}"')
run_command(f'git push origin --tags')
