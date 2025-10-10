#!/usr/bin/env python3

import argparse
import re
import subprocess
import time
import os
import sys

USAGE = """
Use like this:

    gdk_code.py -m <memory> -t <walltime> -a <account> ~/some/dir/on/genomedk

The following needs to be part of your .ssh/config file:

Host gdk
    HostName    login.genome.au.dk
    User        your_user_name_on_genomedk

Host cn-* gn-* s21n* s22n*
    HostName    %h
    ProxyJump   gdk
    User        your_user_name_on_genomedk
"""

def run_command(command):
    """Run a shell command and return the output."""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip(), result.stdout.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()

def run_remote(command, use_ssh=True):
    """Run a command either locally or via ssh gdk."""
    if use_ssh:
        return run_command(f"ssh gdk '{command}'")
    else:
        return run_command(command)

def check_remote_directory(directory, use_ssh=True):
    """
    Check if a directory exists on the remote server and return its expanded path.
    Returns the expanded absolute path if it exists, None otherwise.
    """
    command = f"realpath {directory}"
    if use_ssh:
        command = f"ssh gdk '{command}'"

    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        expanded_path = result.stdout.strip()
        # Verify it's actually a directory
        test_cmd = f"test -d {expanded_path}"
        if use_ssh:
            test_cmd = f"ssh gdk '{test_cmd}'"
        test_result = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if test_result.returncode == 0:
            return expanded_path
    return None

def build_sbatch_command(args):
    """Build sbatch command from provided arguments."""
    cmd_parts = ["sbatch"]

    # Add all sbatch options if they are provided
    if args.account:
        cmd_parts.append(f"--account={args.account}")
    if args.cpus_per_task:
        cmd_parts.append(f"--cpus-per-task={args.cpus_per_task}")
    if args.chdir:
        cmd_parts.append(f"--chdir={args.chdir}")
    if args.error:
        cmd_parts.append(f"--error={args.error}")
    if args.job_name:
        cmd_parts.append(f"--job-name={args.job_name}")
    if args.ntasks:
        cmd_parts.append(f"--ntasks={args.ntasks}")
    if args.no_requeue:
        cmd_parts.append("--no-requeue")
    if args.ntasks_per_node:
        cmd_parts.append(f"--ntasks-per-node={args.ntasks_per_node}")
    if args.nodes:
        cmd_parts.append(f"--nodes={args.nodes}")
    if args.oom_kill_step is not None:
        cmd_parts.append(f"--oom-kill-step={args.oom_kill_step}")
    if args.partition:
        cmd_parts.append(f"--partition={args.partition}")
    if args.requeue:
        cmd_parts.append("--requeue")
    if args.thread_spec:
        cmd_parts.append(f"--thread-spec={args.thread_spec}")
    if args.walltime:
        cmd_parts.append(f"--time={args.walltime}")
    if args.use_min_nodes:
        cmd_parts.append("--use-min-nodes")
    if args.mem:
        cmd_parts.append(f"--mem={args.mem}")
    if args.mincpus:
        cmd_parts.append(f"--mincpus={args.mincpus}")
    if args.nodelist:
        cmd_parts.append(f"--nodelist={args.nodelist}")
    if args.exclude:
        cmd_parts.append(f"--exclude={args.exclude}")
    if args.mem_per_cpu:
        cmd_parts.append(f"--mem-per-cpu={args.mem_per_cpu}")
    if args.cpus_per_gpu:
        cmd_parts.append(f"--cpus-per-gpu={args.cpus_per_gpu}")

    # Add the wrap command
    cmd_parts.append('--wrap="sleep 6d"')

    return " ".join(cmd_parts)

def expand_node_list(nodelist):
    """
    Expand SLURM node list format to individual node names.

    Examples:
        "cn-1041" -> ["cn-1041"]
        "cn-[1041,1053-1055]" -> ["cn-1041", "cn-1053", "cn-1054", "cn-1055"]
        "cn-[1041],gn-[50-52]" -> ["cn-1041", "gn-50", "gn-51", "gn-52"]
    """
    nodes = []

    # Split by comma to handle multiple node groups
    # But we need to be careful about commas inside brackets
    node_groups = []
    current = ""
    bracket_depth = 0

    for char in nodelist:
        if char == '[':
            bracket_depth += 1
            current += char
        elif char == ']':
            bracket_depth -= 1
            current += char
        elif char == ',' and bracket_depth == 0:
            if current.strip():
                node_groups.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        node_groups.append(current.strip())

    # Process each node group
    for group in node_groups:
        # Check if it has bracket notation
        match = re.match(r'^(\S+?)\[([^\]]+)\]$', group)
        if match:
            prefix = match.group(1)
            ranges = match.group(2)

            # Process each range or individual number
            for part in ranges.split(','):
                part = part.strip()
                if '-' in part:
                    # It's a range
                    start, end = part.split('-', 1)
                    for num in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{num}")
                else:
                    # It's a single number
                    nodes.append(f"{prefix}{part}")
        else:
            # No bracket notation, just add as is
            nodes.append(group)

    return nodes

def submit_and_wait_for_job(sbatch_cmd, use_ssh=True):
    """Submit job and wait for it to start running."""
    # Submit job and get job ID
    output = run_remote(sbatch_cmd, use_ssh)
    jobid = output.split()[-1]
    print(f"Slurm job id: {jobid}")

    # Wait for job to start running
    is_running = ""
    while not is_running:
        time.sleep(5)
        is_running = run_remote(f"squeue -j {jobid} -t RUNNING -h", use_ssh)

    time.sleep(10)
    # Get node allocation
    if not use_ssh:
        time.sleep(5)  # Give a moment for the node info to be available on fe-ipsych-01

    nodelist = run_remote(f"squeue -j {jobid} -h -o %N", use_ssh)
    nodes = expand_node_list(nodelist)

    if not nodes:
        print(f"Error: No nodes found for job {jobid}", file=sys.stderr)
        sys.exit(1)

    print(f"Job allocated to node(s): {', '.join(nodes)}")
    return nodes[0]  # Return first node for VSCode connection

def main():
    parser = argparse.ArgumentParser(description="Submit and manage jobs on GenomeDK.")

    # Existing utility arguments
    parser.add_argument("-k", "--kill", help="Kill jobs with a specific name", action="store_true")
    parser.add_argument("-j", "--jobs", help="List pending and running jobs", action="store_true")
    parser.add_argument("directory", nargs="?", help="Directory to open on GenomeDK")

    # Sbatch options
    parser.add_argument("-A", "--account", help="Charge job to specified account")
    parser.add_argument("-c", "--cpus-per-task", help="Number of CPUs required per task", default="1")
    parser.add_argument("-D", "--chdir", help="Set working directory for batch script")
    parser.add_argument("-e", "--error", help="File for batch script's standard error")
    parser.add_argument("-J", "--job-name", help="Name of job")
    parser.add_argument("-n", "--ntasks", help="Number of tasks to run", default="1")
    parser.add_argument("--no-requeue", help="Do not permit the job to be requeued", action="store_true")
    parser.add_argument("--ntasks-per-node", help="Number of tasks to invoke on each node")
    parser.add_argument("-N", "--nodes", help="Number of nodes on which to run (N = min[-max])")
    parser.add_argument("--oom-kill-step", help="Set the OOMKillStep behaviour", type=int, choices=[0, 1])
    parser.add_argument("-p", "--partition", help="Partition requested")
    parser.add_argument("--requeue", help="Permit the job to be requeued", action="store_true")
    parser.add_argument("--thread-spec", help="Count of reserved threads")
    parser.add_argument("-t", "--walltime", help="Time limit (e.g., 08:00:00)", default="08:00:00")
    parser.add_argument("--use-min-nodes", help="If a range of node counts is given, prefer the smaller count", action="store_true")
    parser.add_argument("--mem", help="Minimum amount of real memory (MB)")
    parser.add_argument("--mincpus", help="Minimum number of logical processors per node")
    parser.add_argument("-w", "--nodelist", help="Request a specific list of hosts")
    parser.add_argument("-x", "--exclude", help="Exclude a specific list of hosts")
    parser.add_argument("-m", "--mem-per-cpu", help="Maximum amount of real memory per allocated CPU", default="8g")
    parser.add_argument("--cpus-per-gpu", help="Number of CPUs required per allocated GPU")

    args = parser.parse_args()

    # Handle utility commands
    if args.kill:
        run_command('ssh gdk "scancel --name=wrap"')
        sys.exit(0)

    if args.jobs:
        output = run_command('ssh gdk \'sacct -X --state=PENDING,RUNNING --format="jobid,jobname%30,partition,ReqMem,ReqCPUS,time,elapsed,state"\'')
        print(output)
        sys.exit(0)

    if not args.directory:
        print(USAGE)
        sys.exit(1)

    # Determine if we're on fe-ipsych-01
    hostname = os.getenv("HOSTNAME", "")
    use_ssh = hostname != "fe-ipsych-01"

    # Convert relative path to absolute path with $HOME
    if not args.directory.startswith('/'):
        args.directory = f"$HOME/{args.directory}"

    # Check if directory exists on remote server and get expanded path
    expanded_directory = check_remote_directory(args.directory, use_ssh)
    if not expanded_directory:
        print(f"Error: Directory '{args.directory}' does not exist on the remote server", file=sys.stderr)
        sys.exit(1)

    args.directory = expanded_directory

    # Build and submit job
    sbatch_cmd = build_sbatch_command(args)
    node = submit_and_wait_for_job(sbatch_cmd, use_ssh)

    # Open VSCode with appropriate command
    vscode_uri = f"vscode-remote://ssh-remote+{node}{args.directory}"
    if use_ssh:
        run_command(f"code --folder-uri '{vscode_uri}'")
    else:
        run_command(f"conda run -n $environment code --folder-uri '{vscode_uri}'")

if __name__ == "__main__":
    main()
