#!/usr/bin/env python3
"""
SLURM Script Generator

Generates SLURM submission scripts from configuration files.
This eliminates the need to maintain multiple shell scripts with duplicated boilerplate.

Usage:
    # Generate from config file
    python generate_slurm_script.py --config slurm_configs/production.yaml \\
                                     --script my_script.py \\
                                     --output submit.sh

    # Generate from profile
    python generate_slurm_script.py --profile medium \\
                                     --script my_script.py \\
                                     --output submit.sh

    # Quick submit
    python generate_slurm_script.py --profile small \\
                                     --script my_script.py | sbatch

Author: PtDAlgorithms Team
Date: 2025-10-07
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phasic.cluster_configs import ClusterConfig, load_config, get_default_config


SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#
# Cluster Configuration: {config_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={memory_per_cpu}
#SBATCH --partition={partition}
{qos_line}
{gpu_line}
{extra_options}
{mail_options}

echo "========================================================================"
echo "SLURM Multi-Node Job: {config_name}"
echo "========================================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Nodes:            $SLURM_JOB_NUM_NODES"
echo "CPUs per node:    $SLURM_CPUS_PER_TASK"
echo "Total CPUs:       $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "Total devices:    {total_devices}"
echo "Node list:        $SLURM_JOB_NODELIST"
echo "Started at:       $(date)"
echo "========================================================================"
echo ""

# Load modules
{module_loads}

# Activate Python environment
# Option 1: Pixi (recommended)
eval "$(pixi shell-hook)"

# Option 2: Conda (alternative)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate phasic

# Verify Python and dependencies
echo "Python environment:"
which python
python --version
python -c "import jax; print(f'JAX version: {{jax.__version__}}')" || echo "JAX not found!"
python -c "import phasic; print('PtDAlgorithms imported successfully')" || echo "PtDAlgorithms not found!"
echo ""

# Setup JAX coordinator for distributed computing
COORDINATOR_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_COORDINATOR_ADDRESS=$COORDINATOR_NODE
export JAX_COORDINATOR_PORT={coordinator_port}

echo "JAX Distributed Configuration:"
echo "  Coordinator: $SLURM_COORDINATOR_ADDRESS:$JAX_COORDINATOR_PORT"
echo "  Platform: {platform}"
echo ""

# Set environment variables
{env_vars}

# Run the distributed computation
echo "Starting computation..."
echo ""

srun --kill-on-bad-exit=1 \\
     --cpu-bind=cores \\
     python {script_path}

EXITCODE=$?

echo ""
echo "========================================================================"
echo "Job Completed"
echo "========================================================================"
echo "Exit code: $EXITCODE"
echo "Completed at: $(date)"

if [ $EXITCODE -eq 0 ]; then
    echo "SUCCESS: Job completed successfully"
else
    echo "FAILURE: Job failed with exit code $EXITCODE"
    echo "Check error log: {error_file}"
fi

echo "========================================================================"

exit $EXITCODE
"""


def generate_script(
    config: ClusterConfig,
    script_path: str,
    output_path: Optional[str] = None,
    job_name: Optional[str] = None
) -> str:
    """
    Generate SLURM submission script from configuration.

    Parameters
    ----------
    config : ClusterConfig
        Cluster configuration
    script_path : str
        Path to Python script to execute
    output_path : str, optional
        Path to save generated script. If None, prints to stdout.
    job_name : str, optional
        SLURM job name. If None, derives from script name.

    Returns
    -------
    str
        Generated SLURM script content
    """
    # Determine job name
    if job_name is None:
        job_name = Path(script_path).stem

    # Build QoS line
    qos_line = f"#SBATCH --qos={config.qos}" if config.qos else ""

    # Build GPU line
    gpu_line = ""
    if config.platform == "gpu" and config.gpus_per_node:
        gpu_line = f"#SBATCH --gres=gpu:{config.gpus_per_node}"

    # Build extra SBATCH options
    extra_options = []
    for key, value in config.extra_sbatch_options.items():
        if isinstance(value, bool):
            if value:
                extra_options.append(f"#SBATCH --{key}")
        else:
            extra_options.append(f"#SBATCH --{key}={value}")
    extra_options_str = "\n".join(extra_options) if extra_options else ""

    # Build mail options (if not in extra_options)
    mail_options = ""
    if "mail-type" not in config.extra_sbatch_options:
        mail_options = "# Uncomment for email notifications:\n"
        mail_options += "##SBATCH --mail-type=BEGIN,END,FAIL\n"
        mail_options += "##SBATCH --mail-user=your.email@institution.edu"

    # Build module loads
    if config.modules_to_load:
        module_loads = "\n".join(f"module load {mod}" for mod in config.modules_to_load)
        module_loads = f"echo \"Loading modules...\"\n{module_loads}\necho \"\""
    else:
        module_loads = "# No modules to load"

    # Build environment variables
    env_vars_list = []
    for key, value in config.env_vars.items():
        env_vars_list.append(f"export {key}=\"{value}\"")

    # Add network configuration if specified
    if config.network_interface:
        env_vars_list.append(f"export NCCL_SOCKET_IFNAME=\"{config.network_interface}\"")

    env_vars_str = "\n".join(env_vars_list) if env_vars_list else "# No environment variables"

    # Format script
    script_content = SLURM_SCRIPT_TEMPLATE.format(
        job_name=job_name,
        output_file=f"logs/{job_name}_%j.out",
        error_file=f"logs/{job_name}_%j.err",
        config_name=config.name,
        nodes=config.nodes,
        cpus_per_node=config.cpus_per_node,
        time_limit=config.time_limit,
        memory_per_cpu=config.memory_per_cpu,
        partition=config.partition,
        qos_line=qos_line,
        gpu_line=gpu_line,
        extra_options=extra_options_str,
        mail_options=mail_options,
        total_devices=config.total_devices,
        module_loads=module_loads,
        coordinator_port=config.coordinator_port,
        platform=config.platform,
        env_vars=env_vars_str,
        script_path=script_path,
    )

    # Save or print
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(script_content)
        print(f"Generated SLURM script: {output_file}", file=sys.stderr)

        # Make executable
        output_file.chmod(0o755)
    else:
        # Print to stdout (can pipe to sbatch)
        print(script_content)

    return script_content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM submission scripts from configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from config file
    %(prog)s --config slurm_configs/production.yaml --script my_script.py --output submit.sh

    # Generate from profile
    %(prog)s --profile medium --script my_script.py --output submit.sh

    # Quick submit (pipe to sbatch)
    %(prog)s --profile small --script my_script.py | sbatch

    # Show available profiles
    %(prog)s --list-profiles
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--profile', '-p',
        type=str,
        help='Use predefined profile (debug, small, medium, large, production)'
    )

    parser.add_argument(
        '--script', '-s',
        type=str,
        help='Path to Python script to execute'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for generated script. If not specified, prints to stdout.'
    )

    parser.add_argument(
        '--job-name', '-n',
        type=str,
        help='SLURM job name. If not specified, derived from script name.'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available configuration profiles'
    )

    args = parser.parse_args()

    # List profiles
    if args.list_profiles:
        print("Available configuration profiles:")
        for profile in ["debug", "small", "medium", "large", "production"]:
            config = get_default_config(profile)
            print(f"\n{profile}:")
            print(f"  Nodes: {config.nodes}")
            print(f"  CPUs/node: {config.cpus_per_node}")
            print(f"  Total devices: {config.total_devices}")
            print(f"  Time limit: {config.time_limit}")
        return 0

    # Validate arguments
    if not args.script:
        parser.error("--script is required")

    if not args.config and not args.profile:
        parser.error("Either --config or --profile is required")

    if args.config and args.profile:
        parser.error("Cannot specify both --config and --profile")

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config(args.profile)

    # Generate script
    generate_script(
        config=config,
        script_path=args.script,
        output_path=args.output,
        job_name=args.job_name
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
