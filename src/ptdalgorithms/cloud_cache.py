"""
Cloud Storage Integration for Cache Distribution

Provides backends for storing and retrieving cached symbolic DAGs from
cloud storage services (S3, GCS, Azure) and HTTP sources.

This enables:
- Centralized model repositories
- Team/organization cache sharing
- Public model libraries
- Automatic cache synchronization

Example Usage:
    >>> from ptdalgorithms.cloud_cache import S3Backend, download_from_url
    >>>
    >>> # Download from public HTTP source
    >>> download_from_url(
    ...     'https://github.com/org/models/releases/download/v1/cache.tar.gz',
    ...     extract_to='~/.ptdalgorithms_cache/symbolic'
    ... )
    >>>
    >>> # Use S3 backend
    >>> backend = S3Backend('my-bucket', 'ptdalgorithms/cache/')
    >>> backend.download_cache()
"""

import os
import json
import tarfile
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import shutil

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


class CloudBackend:
    """
    Abstract base class for cloud storage backends.

    Subclasses should implement:
    - upload_file(local_path, remote_key)
    - download_file(remote_key, local_path)
    - list_files(prefix)
    - delete_file(remote_key)
    """

    def upload_cache(
        self,
        local_cache_dir: Union[Path, str],
        remote_prefix: str = '',
        include_patterns: Optional[List[str]] = None
    ):
        """
        Upload cache directory to cloud storage.

        Parameters
        ----------
        local_cache_dir : Path or str
            Local cache directory to upload
        remote_prefix : str, optional
            Prefix for remote keys. Default: ''
        include_patterns : list of str, optional
            Glob patterns to include. Default: all files
        """
        local_cache_dir = Path(local_cache_dir)

        if not local_cache_dir.exists():
            raise ValueError(f"Cache directory does not exist: {local_cache_dir}")

        # Collect files to upload
        if include_patterns is None:
            files = list(local_cache_dir.rglob('*'))
        else:
            files = []
            for pattern in include_patterns:
                files.extend(local_cache_dir.glob(pattern))

        files = [f for f in files if f.is_file()]

        print(f"Uploading {len(files)} files to cloud storage...")

        for i, file_path in enumerate(files, 1):
            rel_path = file_path.relative_to(local_cache_dir)
            remote_key = f"{remote_prefix}{rel_path}".lstrip('/')

            print(f"  [{i}/{len(files)}] {rel_path}...", end=' ', flush=True)
            self.upload_file(file_path, remote_key)
            print("✓")

        print(f"✓ Upload complete: {len(files)} files")

    def download_cache(
        self,
        local_cache_dir: Union[Path, str],
        remote_prefix: str = '',
        overwrite: bool = False
    ):
        """
        Download cache from cloud storage.

        Parameters
        ----------
        local_cache_dir : Path or str
            Local directory to download to
        remote_prefix : str, optional
            Prefix for remote keys. Default: ''
        overwrite : bool, optional
            Overwrite existing local files. Default: False
        """
        local_cache_dir = Path(local_cache_dir)
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        # List remote files
        remote_files = self.list_files(remote_prefix)

        print(f"Downloading {len(remote_files)} files from cloud storage...")

        downloaded = 0
        skipped = 0

        for i, remote_key in enumerate(remote_files, 1):
            # Determine local path
            rel_path = remote_key[len(remote_prefix):].lstrip('/')
            local_path = local_cache_dir / rel_path

            if local_path.exists() and not overwrite:
                skipped += 1
                continue

            print(f"  [{i}/{len(remote_files)}] {rel_path}...", end=' ', flush=True)

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.download_file(remote_key, local_path)
            downloaded += 1
            print("✓")

        print(f"✓ Download complete: {downloaded} downloaded, {skipped} skipped")

    def upload_file(self, local_path: Path, remote_key: str):
        """Upload single file to cloud storage"""
        raise NotImplementedError("Subclass must implement upload_file()")

    def download_file(self, remote_key: str, local_path: Path):
        """Download single file from cloud storage"""
        raise NotImplementedError("Subclass must implement download_file()")

    def list_files(self, prefix: str = '') -> List[str]:
        """List files in cloud storage with given prefix"""
        raise NotImplementedError("Subclass must implement list_files()")

    def delete_file(self, remote_key: str):
        """Delete file from cloud storage"""
        raise NotImplementedError("Subclass must implement delete_file()")


class S3Backend(CloudBackend):
    """
    AWS S3 backend for cache storage.

    Requires: boto3

    Parameters
    ----------
    bucket_name : str
        S3 bucket name
    prefix : str, optional
        Prefix within bucket. Default: ''
    region : str, optional
        AWS region. Default: None (use default)
    profile : str, optional
        AWS profile name. Default: None

    Examples
    --------
    >>> backend = S3Backend('my-bucket', 'ptdalgorithms/cache/')
    >>> backend.upload_cache('~/.ptdalgorithms_cache/symbolic')
    >>> backend.download_cache('/tmp/cache')
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = '',
        region: Optional[str] = None,
        profile: Optional[str] = None
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 required for S3 backend. Install with: pip install boto3"
            )

        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''

        # Create S3 client
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region

        session = boto3.Session(**session_kwargs)
        self.s3 = session.client('s3')

    def upload_file(self, local_path: Path, remote_key: str):
        """Upload file to S3"""
        self.s3.upload_file(str(local_path), self.bucket_name, remote_key)

    def download_file(self, remote_key: str, local_path: Path):
        """Download file from S3"""
        self.s3.download_file(self.bucket_name, remote_key, str(local_path))

    def list_files(self, prefix: str = '') -> List[str]:
        """List files in S3 bucket"""
        full_prefix = self.prefix + prefix

        keys = []
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    keys.append(obj['Key'])

        return keys

    def delete_file(self, remote_key: str):
        """Delete file from S3"""
        self.s3.delete_object(Bucket=self.bucket_name, Key=remote_key)


class GCSBackend(CloudBackend):
    """
    Google Cloud Storage backend for cache storage.

    Requires: google-cloud-storage

    Parameters
    ----------
    bucket_name : str
        GCS bucket name
    prefix : str, optional
        Prefix within bucket. Default: ''
    project : str, optional
        GCP project ID. Default: None
    credentials_path : str, optional
        Path to service account JSON. Default: None (use default credentials)

    Examples
    --------
    >>> backend = GCSBackend('my-bucket', 'ptdalgorithms/cache/')
    >>> backend.upload_cache('~/.ptdalgorithms_cache/symbolic')
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = '',
        project: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            )

        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''

        # Create GCS client
        if credentials_path:
            self.client = storage.Client.from_service_account_json(
                credentials_path, project=project
            )
        else:
            self.client = storage.Client(project=project)

        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: Path, remote_key: str):
        """Upload file to GCS"""
        blob = self.bucket.blob(remote_key)
        blob.upload_from_filename(str(local_path))

    def download_file(self, remote_key: str, local_path: Path):
        """Download file from GCS"""
        blob = self.bucket.blob(remote_key)
        blob.download_to_filename(str(local_path))

    def list_files(self, prefix: str = '') -> List[str]:
        """List files in GCS bucket"""
        full_prefix = self.prefix + prefix

        blobs = self.client.list_blobs(self.bucket_name, prefix=full_prefix)
        return [blob.name for blob in blobs]

    def delete_file(self, remote_key: str):
        """Delete file from GCS"""
        blob = self.bucket.blob(remote_key)
        blob.delete()


class AzureBlobBackend(CloudBackend):
    """
    Azure Blob Storage backend for cache storage.

    Requires: azure-storage-blob

    Parameters
    ----------
    account_name : str
        Azure storage account name
    container_name : str
        Container name
    prefix : str, optional
        Prefix within container. Default: ''
    account_key : str, optional
        Account key. Default: None (use default credentials)
    connection_string : str, optional
        Connection string. Default: None

    Examples
    --------
    >>> backend = AzureBlobBackend(
    ...     account_name='mystorageaccount',
    ...     container_name='ptdalgorithms',
    ...     account_key='...'
    ... )
    >>> backend.upload_cache('~/.ptdalgorithms_cache/symbolic')
    """

    def __init__(
        self,
        account_name: str,
        container_name: str,
        prefix: str = '',
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob required for Azure backend. "
                "Install with: pip install azure-storage-blob"
            )

        self.container_name = container_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''

        # Create blob service client
        if connection_string:
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        elif account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(account_url, credential=account_key)
        else:
            # Use default credentials
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(account_url)

        self.container_client = self.blob_service.get_container_client(container_name)

    def upload_file(self, local_path: Path, remote_key: str):
        """Upload file to Azure Blob Storage"""
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=remote_key
        )

        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_file(self, remote_key: str, local_path: Path):
        """Download file from Azure Blob Storage"""
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=remote_key
        )

        with open(local_path, 'wb') as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

    def list_files(self, prefix: str = '') -> List[str]:
        """List blobs in container"""
        full_prefix = self.prefix + prefix

        blob_list = self.container_client.list_blobs(name_starts_with=full_prefix)
        return [blob.name for blob in blob_list]

    def delete_file(self, remote_key: str):
        """Delete blob from container"""
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=remote_key
        )
        blob_client.delete_blob()


# ============================================================================
# HTTP/HTTPS Download Support
# ============================================================================

def download_from_url(
    url: str,
    output_path: Optional[Union[Path, str]] = None,
    extract_to: Optional[Union[Path, str]] = None,
    verify_checksum: Optional[str] = None,
    show_progress: bool = True
) -> Path:
    """
    Download cache archive from HTTP/HTTPS URL.

    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path or str, optional
        Where to save downloaded file. Default: temporary file
    extract_to : Path or str, optional
        Directory to extract archive to. If None, doesn't extract.
    verify_checksum : str, optional
        Expected SHA-256 checksum to verify. Default: None (no verification)
    show_progress : bool, optional
        Show download progress. Default: True

    Returns
    -------
    Path
        Path to downloaded file (or extraction directory if extracted)

    Examples
    --------
    >>> # Download and extract
    >>> download_from_url(
    ...     'https://example.com/models/cache_v1.tar.gz',
    ...     extract_to='~/.ptdalgorithms_cache/symbolic'
    ... )
    >>>
    >>> # Download with checksum verification
    >>> download_from_url(
    ...     'https://example.com/models/cache_v1.tar.gz',
    ...     verify_checksum='a3f2e9c8b1d4...',
    ...     extract_to='~/.ptdalgorithms_cache/symbolic'
    ... )
    """
    if not HAS_REQUESTS:
        raise ImportError("requests required for HTTP downloads. Install with: pip install requests")

    # Determine output path
    if output_path is None:
        # Use temporary file
        suffix = Path(url).suffix or '.tar.gz'
        temp_fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(temp_fd)
        output_path = Path(output_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading from {url}")
    print(f"  Saving to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0

    hasher = hashlib.sha256() if verify_checksum else None

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if hasher:
                    hasher.update(chunk)

                if show_progress and total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"  Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)",
                          end='\r', flush=True)

    if show_progress:
        print()  # New line after progress

    print(f"✓ Download complete: {downloaded / 1024 / 1024:.1f} MB")

    # Verify checksum
    if verify_checksum:
        actual_checksum = hasher.hexdigest()
        if actual_checksum != verify_checksum:
            output_path.unlink()
            raise ValueError(
                f"Checksum mismatch!\n"
                f"  Expected: {verify_checksum}\n"
                f"  Actual:   {actual_checksum}"
            )
        print(f"✓ Checksum verified: {actual_checksum[:16]}...")

    # Extract if requested
    if extract_to:
        extract_to = Path(extract_to).expanduser()
        extract_to.mkdir(parents=True, exist_ok=True)

        print(f"Extracting to {extract_to}...")

        with tarfile.open(output_path, 'r:*') as tar:
            tar.extractall(extract_to)

        print(f"✓ Extraction complete")

        # Clean up downloaded file
        output_path.unlink()

        return extract_to

    return output_path


def download_from_github_release(
    repo: str,
    tag: str,
    asset_name: str,
    extract_to: Optional[Union[Path, str]] = None,
    verify_checksum: Optional[str] = None
) -> Path:
    """
    Download cache archive from GitHub release.

    Parameters
    ----------
    repo : str
        Repository in format 'owner/repo'
    tag : str
        Release tag (e.g., 'v1.0.0')
    asset_name : str
        Name of release asset to download
    extract_to : Path or str, optional
        Directory to extract to. Default: None (don't extract)
    verify_checksum : str, optional
        Expected SHA-256 checksum. Default: None

    Returns
    -------
    Path
        Path to downloaded/extracted files

    Examples
    --------
    >>> # Download from GitHub release
    >>> download_from_github_release(
    ...     repo='munch-group/ptdalgorithms-models',
    ...     tag='v1.0.0',
    ...     asset_name='coalescent_models.tar.gz',
    ...     extract_to='~/.ptdalgorithms_cache/symbolic'
    ... )
    """
    if not HAS_REQUESTS:
        raise ImportError("requests required. Install with: pip install requests")

    # Get release info from GitHub API
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    print(f"Fetching release info for {repo}@{tag}...")
    response = requests.get(api_url)
    response.raise_for_status()

    release_info = response.json()

    # Find asset
    asset = None
    for a in release_info.get('assets', []):
        if a['name'] == asset_name:
            asset = a
            break

    if asset is None:
        available = [a['name'] for a in release_info.get('assets', [])]
        raise ValueError(
            f"Asset '{asset_name}' not found in release.\n"
            f"Available assets: {', '.join(available)}"
        )

    # Download asset
    download_url = asset['browser_download_url']

    return download_from_url(
        url=download_url,
        extract_to=extract_to,
        verify_checksum=verify_checksum,
        show_progress=True
    )


def install_model_library(
    library_name: str,
    cache_dir: Optional[Union[Path, str]] = None
):
    """
    Install pre-computed model library from official repository.

    Parameters
    ----------
    library_name : str
        Name of model library to install. Available libraries:
        - 'coalescent' - Population genetics coalescent models
        - 'queuing' - Queuing theory models
        - 'reliability' - Reliability and survival models
    cache_dir : Path or str, optional
        Cache directory to install to. Default: ~/.ptdalgorithms_cache/symbolic

    Examples
    --------
    >>> # Install official coalescent models library
    >>> install_model_library('coalescent')
    >>> # Models are now cached and ready to use!
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.ptdalgorithms_cache' / 'symbolic'
    else:
        cache_dir = Path(cache_dir)

    # Map library names to GitHub release info
    libraries = {
        'coalescent': {
            'repo': 'munch-group/ptdalgorithms-models',
            'tag': 'v1.0.0',
            'asset': 'coalescent_models.tar.gz'
        },
        'queuing': {
            'repo': 'munch-group/ptdalgorithms-models',
            'tag': 'v1.0.0',
            'asset': 'queuing_models.tar.gz'
        },
        'reliability': {
            'repo': 'munch-group/ptdalgorithms-models',
            'tag': 'v1.0.0',
            'asset': 'reliability_models.tar.gz'
        }
    }

    if library_name not in libraries:
        available = ', '.join(libraries.keys())
        raise ValueError(
            f"Unknown library '{library_name}'.\n"
            f"Available libraries: {available}"
        )

    info = libraries[library_name]

    print(f"Installing model library: {library_name}")

    # Download and extract
    download_from_github_release(
        repo=info['repo'],
        tag=info['tag'],
        asset_name=info['asset'],
        extract_to=cache_dir
    )

    print(f"✓ Library '{library_name}' installed successfully!")
    print(f"  Location: {cache_dir}")
