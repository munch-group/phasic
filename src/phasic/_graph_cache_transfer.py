"""Registry cache pull/push functions for :class:`~phasic.Graph`.

Extracted verbatim from ``Graph`` (Stage-3 WS-C). Pure relocation; the bodies
are unchanged (lazy local imports). Assigned onto ``Graph`` as class attributes
in ``__init__.py`` so they stay direct members (docs/introspection).
"""
from __future__ import annotations


def pull_cache(self, force: bool = False) -> bool:
    """Download the published compute artifact for this graph if available.

    Looks up this graph's content hash in the ``munch-group/phasic-traces``
    registry. On a hit, downloads the parent ``.bin`` (and any per-SCC
    files) to ``~/.phasic_cache/parameterized_reward_compute/`` so the
    next call to :meth:`expectation` / :meth:`pdf` / :meth:`moments`
    reuses the published elimination instead of recomputing.

    No git, gh, or daemon required for the consumer side — fetches are
    plain HTTPS to ``raw.githubusercontent.com``.

    Parameters
    ----------
    force : bool, default False
        If True, re-download even when a local cache file already exists.

    Returns
    -------
    bool
        True if a fresh download succeeded or a local cache file was
        already present. False if no registry entry matches this
        graph's content hash.

    Raises
    ------
    phasic.exceptions.PTDBackendError
        On network failure or SHA-256 mismatch.
    phasic.exceptions.PTDFormatError
        If the artifact's ``format_revision`` exceeds what this
        phasic build supports.

    Examples
    --------
    >>> g = phasic.Graph(my_callback, ipv=[5])
    >>> if g.pull_cache():
    ...     print('reusing published elimination')
    >>> e = g.expectation()
    """
    from .compute_repository import _get_default_registry
    return _get_default_registry().pull(self, force=force)

def push_cache(
    self,
    *,
    id: str,
    description: str,
    domain: str | None = None,
    model_type: str | None = None,
    tags: list[str] | None = None,
    license: str = 'MIT',
    author: str | None = None,
    registry_repo: str = 'munch-group/phasic-traces',
    dry_run: bool = False,
    overwrite_branch: bool = False,
) -> str:
    """Publish this graph's compute artifact to the phasic-traces registry.

    Populates the C-side elimination cache if necessary (calls
    :meth:`expectation`), saves the parent ``.bin`` and any per-SCC
    files, clones the registry repo to a temporary directory, splices
    in a new entry, pushes a feature branch, and opens a pull request
    via ``gh``.

    Parameters
    ----------
    id : str
        Human-readable identifier for the registry entry
        (e.g. ``'coal_n5_theta1'``). Must be unique within the
        registry.
    description : str
        One-line free-form description of the model.
    domain, model_type : str, optional
        Filtering keys stored in the entry's metadata.
    tags : list[str], optional
        Tags stored in the entry's metadata.
    license : str, default 'MIT'
        SPDX license identifier.
    author : str, optional
        Override; default is ``"Name <email>"`` from
        ``git config user.{name,email}``.
    registry_repo : str, default 'munch-group/phasic-traces'
        GitHub ``owner/name`` of the registry repository.
    dry_run : bool, default False
        If True, build the artifacts and return a JSON string of the
        would-be entry without cloning or pushing.
    overwrite_branch : bool, default False
        If a stale ``phasic-publish/<id>`` branch already exists on
        the remote (e.g. from an earlier failed push) the call
        refuses unless this is ``True``. With ``True``, the push
        uses ``--force-with-lease``: it overwrites the stale branch
        but still refuses if a third party has pushed concurrently.

    Returns
    -------
    str
        URL of the opened PR (or, if ``dry_run=True``, a JSON string).

    Raises
    ------
    phasic.exceptions.PTDBackendError
        If ``gh`` is missing or unauthenticated, the entry id already
        exists in ``registry.json``, the feature branch already
        exists on the remote (without ``overwrite_branch=True``),
        or git/gh fails.

    Notes
    -----
    Running ``gh auth login`` once in a terminal sets up authentication
    for all future ``push_cache()`` calls. ``push_cache`` does not
    prompt inside the notebook.

    Examples
    --------
    >>> g = phasic.Graph(my_callback, ipv=[5])
    >>> g.expectation()  # populate C-side cache (optional; push_cache does this)
    >>> entry_json = g.push_cache(
    ...     id='my_model_v1',
    ...     description='Kingman coalescent for n=5',
    ...     domain='population-genetics',
    ...     model_type='coalescent',
    ...     dry_run=True,
    ... )
    """
    from .compute_repository import ComputeRegistry, _default_author
    metadata: dict = {
        "description": description,
        "author": author or _default_author(),
        "license": license,
        "vertices": self.vertices_length(),
        "param_length": self.param_length(),
    }
    if domain:
        metadata["domain"] = domain
    if model_type:
        metadata["model_type"] = model_type
    if tags:
        metadata["tags"] = list(tags)

    registry = ComputeRegistry(
        registry_repo=registry_repo, auto_update=False)
    return registry.push(
        self, compute_id=id, metadata=metadata,
        dry_run=dry_run, overwrite_branch=overwrite_branch,
    )
