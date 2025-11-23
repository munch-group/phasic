"""Command-line interface for hybrid P2P distribution."""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .client import Client, ClientConfig
from .signing import generate_keypair
from .validation import ContentManifest

# Setup rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Hybrid P2P content distribution tool."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--name", "-n", required=True, help="Package name")
@click.option("--version", "-V", required=True, help="Semantic version")
@click.option("--uploader-id", "-u", required=True, help="Uploader identifier")
@click.option("--description", "-d", help="Package description")
@click.option("--key", "-k", type=click.Path(exists=True, path_type=Path), help="Signing key path")
@click.option("--no-torrent", is_flag=True, help="Skip torrent creation")
@click.option("--no-pin", is_flag=True, help="Don't pin to IPFS")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output manifest path")
def publish(
    files: tuple[Path, ...],
    name: str,
    version: str,
    uploader_id: str,
    description: Optional[str],
    key: Optional[Path],
    no_torrent: bool,
    no_pin: bool,
    output: Optional[Path],
) -> None:
    """
    Publish files to P2P networks.
    
    Example:
        hybrid-p2p publish file1.txt file2.txt -n mypackage -V 1.0.0 -u alice
    """
    if not files:
        console.print("[red]Error:[/red] No files specified")
        sys.exit(1)
    
    try:
        config = ClientConfig(
            signing_key_path=key,
        )
        
        with Client(config) as client:
            with console.status("[bold green]Publishing..."):
                manifest = client.publish(
                    files=list(files),
                    name=name,
                    version=version,
                    uploader_id=uploader_id,
                    description=description,
                    create_torrent=not no_torrent,
                    pin_to_ipfs=not no_pin,
                )
            
            # Display results
            console.print("\n[bold green]✓ Published successfully![/bold green]\n")
            
            table = Table(title="Publication Details")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Name", manifest.name)
            table.add_row("Version", manifest.version)
            table.add_row("Content ID", manifest.content_id)
            table.add_row("Total Size", f"{manifest.total_size:,} bytes")
            table.add_row("Files", str(len(manifest.files)))
            
            if manifest.distribution:
                if manifest.distribution.ipfs_cid:
                    table.add_row("IPFS CID", manifest.distribution.ipfs_cid)
                if manifest.distribution.torrent_infohash:
                    table.add_row("Torrent Hash", manifest.distribution.torrent_infohash)
            
            console.print(table)
            
            # Save manifest
            if output:
                output.write_text(manifest.model_dump_json(indent=2))
                console.print(f"\n[green]Manifest saved to:[/green] {output}")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if logging.getLogger().level == logging.DEBUG:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--no-verify", is_flag=True, help="Skip signature verification")
@click.option("--prefer-bt", is_flag=True, help="Prefer BitTorrent over IPFS")
def fetch(
    manifest: Path,
    output: Optional[Path],
    no_verify: bool,
    prefer_bt: bool,
) -> None:
    """
    Fetch content from P2P networks using a manifest.
    
    Example:
        hybrid-p2p fetch manifest.json -o ./downloads
    """
    try:
        config = ClientConfig()
        
        with Client(config) as client:
            with console.status("[bold green]Fetching..."):
                output_dir = client.fetch(
                    manifest=manifest,
                    output_dir=output,
                    verify_signature=not no_verify,
                    prefer_ipfs=not prefer_bt,
                )
            
            # Load manifest for display
            m = ContentManifest.model_validate_json(manifest.read_text())
            
            console.print("\n[bold green]✓ Fetched successfully![/bold green]\n")
            console.print(f"[green]Files saved to:[/green] {output_dir}\n")
            
            # List files
            table = Table(title="Downloaded Files")
            table.add_column("File", style="cyan")
            table.add_column("Size", style="white", justify="right")
            table.add_column("Type", style="yellow")
            
            for file_entry in m.files:
                table.add_row(
                    file_entry.path,
                    f"{file_entry.size:,}",
                    file_entry.mime_type,
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if logging.getLogger().level == logging.DEBUG:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.argument("content_dir", type=click.Path(exists=True, path_type=Path))
def verify(manifest: Path, content_dir: Path) -> None:
    """
    Verify local content against a manifest.
    
    Example:
        hybrid-p2p verify manifest.json ./content
    """
    try:
        config = ClientConfig()
        
        with Client(config) as client:
            with console.status("[bold green]Verifying..."):
                client.verify_local_content(manifest, content_dir)
            
            console.print("\n[bold green]✓ All files verified successfully![/bold green]")
            
    except Exception as e:
        console.print(f"[red]✗ Verification failed:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--name", "-n", default="signing_key", help="Key name prefix")
@click.option("--password", "-p", is_flag=True, help="Encrypt with password")
def keygen(output: Optional[Path], name: str, password: bool) -> None:
    """
    Generate a new Ed25519 signing key pair.
    
    Example:
        hybrid-p2p keygen -o ~/.hybrid_p2p/keys -n mykey
    """
    output_dir = output or Path.cwd()
    
    pwd = None
    if password:
        pwd = click.prompt("Enter password", hide_input=True, confirmation_prompt=True)
        pwd = pwd.encode("utf-8")
    
    try:
        private_path, public_path = generate_keypair(
            output_dir=output_dir,
            key_name=name,
            password=pwd,
        )
        
        console.print("\n[bold green]✓ Key pair generated![/bold green]\n")
        console.print(f"[green]Private key:[/green] {private_path}")
        console.print(f"[green]Public key:[/green] {public_path}")
        console.print("\n[yellow]Keep the private key secure![/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
def inspect(manifest: Path) -> None:
    """
    Inspect a manifest file.
    
    Example:
        hybrid-p2p inspect manifest.json
    """
    try:
        m = ContentManifest.model_validate_json(manifest.read_text())
        
        # Basic info
        table = Table(title="Manifest Information")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Name", m.name)
        table.add_row("Version", m.version)
        table.add_row("Content ID", m.content_id)
        table.add_row("Total Size", f"{m.total_size:,} bytes")
        table.add_row("Files", str(len(m.files)))
        
        if m.description:
            table.add_row("Description", m.description)
        
        console.print(table)
        console.print()
        
        # Files
        files_table = Table(title="Files")
        files_table.add_column("Path", style="cyan")
        files_table.add_column("Size", style="white", justify="right")
        files_table.add_column("SHA-256", style="yellow")
        
        for file_entry in m.files:
            files_table.add_row(
                file_entry.path,
                f"{file_entry.size:,}",
                file_entry.sha256[:16] + "...",
            )
        
        console.print(files_table)
        console.print()
        
        # Distribution
        if m.distribution:
            dist_table = Table(title="Distribution")
            dist_table.add_column("Network", style="cyan")
            dist_table.add_column("Identifier", style="white")
            
            if m.distribution.ipfs_cid:
                dist_table.add_row("IPFS", m.distribution.ipfs_cid)
            if m.distribution.torrent_infohash:
                dist_table.add_row("BitTorrent", m.distribution.torrent_infohash)
            
            console.print(dist_table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
