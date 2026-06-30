#!/usr/bin/env python3
"""Generate the phasic C/C++ API reference (Quarto ``.qmd``) from Doxygen XML.

Pipeline (see ``cpp-api-reference-plan.md``)::

    doxygen docs/Doxyfile          ->  docs/_doxygen/xml/*.xml
    python scripts/gen_cpp_api.py  ->  docs/cpp_api/*.qmd + index.qmd + _sidebar.yml

The C++ class prose is enriched by reusing the shared Python/pybind docstrings
(the Python ``Graph`` *is* the C++ ``phasic::Graph``); C-API prose comes from the
Doxygen comments in ``api/c/phasic.h``. Output mirrors the quartodoc-generated
``docs/api/*.qmd`` so the two reference trees look identical.

Run from anywhere; paths are resolved relative to this file.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Docstring reuse is part of the chosen design, so a failed import is a hard
# error (no silent fallback) -- the generator must run where phasic is installed.
import phasic

REPO = Path(__file__).resolve().parent.parent
XML = REPO / "docs" / "_doxygen" / "xml"
OUT = REPO / "docs" / "cpp_api"

# --- C++ classes to document, in display order (rf_graph is internal) --------
CPP_CLASS_ORDER = [
    "Graph",
    "Vertex",
    "Edge",
    "ParameterizedEdge",
    "PhaseTypeDistribution",
    "ProbabilityDistributionContext",
    "DPHProbabilityDistributionContext",
    "AnyProbabilityDistributionContext",
    "SCCGraph",
    "SCCVertex",
]

# Only the Python ``Graph`` is rendered as its own quartodoc page, so it is the
# only class we cross-link to (../api/Graph.qmd) and reuse docstrings from.
PY_DOC_CLASS = "Graph"

# C++ method name -> Python attribute name (pybind renames; everything else is
# assumed identical and looked up directly on phasic.Graph).
RENAME = {
    "dph_pmf": "pdf_discrete",
    "dph_cdf": "cdf_discrete",
    "dph_reward_transform": "reward_transform_discrete",
    "dph_stop_probability": "stop_probability_discrete",
    "dph_normalize": "normalize_discrete",
    "random_sample": "sample",
    "dph_random_sample": "sample_discrete",
    "mph_random_sample": "sample_multivariate",
    "mdph_random_sample": "sample_multivariate",
}

# --- C-API category pages: (slug, title, description, [name-substrings]) ------
# First match wins, so order from most specific to most generic.
C_CATEGORIES = [
    ("c_avl", "AVL trees",
     "Balanced binary search trees used to index vertices by their state vector.",
     ["avl"]),
    ("c_symbolic", "Symbolic expression system",
     "The symbolic expression DAG underlying parameterized graph elimination.",
     ["expr", "graph_symbolic"]),
    ("c_trace", "Trace-based elimination",
     "Record-once / evaluate-many elimination traces and their on-disk caching.",
     ["trace"]),
    ("c_scc", "Strongly connected components",
     "SCC decomposition, synthetic-graph construction, and parallel PRC composition.",
     ["scc", "strongly_connected", "synth"]),
    ("c_cache", "Reward-compute graph & on-disk cache",
     "Symbolic reward-compute graphs and their persistent (rev-3) parameterized cache.",
     ["reward_compute", "save_parameterized", "load_parameterized",
      "pcg_rev3", "cache_root", "ex_absorbation", "precompute"]),
    ("c_weights", "Parameterized weights & formula tape",
     "Updating edge weights / the initial probability vector, and the per-edge "
     "weight-formula bytecode tape.",
     ["graph_update_weights", "update_ipv", "param_length", "weight_tape"]),
    ("c_sampling", "Random sampling",
     "Forward sampling of absorption times, jumps, sample paths, and stop vertices.",
     ["random_sample", "sample_path", "backward_prob"]),
    ("c_distributions", "Distributions & forward contexts",
     "PDF/CDF/PMF, Laplace transform, normalization, and the forward-stepping "
     "probability-distribution contexts.",
     ["pdf", "cdf", "pmf", "laplace", "probability_distribution",
      "normalize", "phase_type_distribution"]),
    ("c_moments", "Moments, sojourn & waiting time",
     "Expected waiting time, expected sojourn times, and graph defect.",
     ["expected_", "sojourn", "waiting", "defect", "moment"]),
    ("c_rewards", "Reward transforms",
     "Reward transformation of continuous and discrete phase-type graphs.",
     ["reward_transform"]),
    ("c_graph", "Graph construction & core",
     "Creating graphs, vertices and edges; cloning, validation, and topology.",
     ["graph", "vertex", "edge", "directed", "clone", "acyclic",
      "validate", "topological", "notify", "find_or_create"]),
]
C_MISC = ("c_misc", "Other entry points", "Additional C functions.")


# --------------------------------------------------------------------------- #
# XML / markdown helpers
# --------------------------------------------------------------------------- #
def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def flatten_text(elem) -> str:
    return _ws("".join(elem.itertext())) if elem is not None else ""


def render_inline(ch) -> str:
    tag = ch.tag
    inner = "".join(ch.itertext())
    if tag == "computeroutput":
        return f"`{inner}`"
    if tag == "bold":
        return f"**{inner}**"
    if tag in ("emphasis", "italic"):
        return f"*{inner}*"
    if tag == "ulink":
        return f"[{inner}]({ch.get('url', '')})"
    if tag == "linebreak":
        return "  \n"
    # ref, sp, anchor, formula, ndash, mdash, ... -> plain text
    return inner


def render_paramlist(pl) -> str:
    label = {
        "param": "Parameters", "retval": "Return values",
        "exception": "Exceptions", "templateparam": "Template parameters",
    }.get(pl.get("kind"), "Parameters")
    lines = [f"**{label}:**", ""]
    for item in pl.findall("parameteritem"):
        names = [_ws("".join(n.itertext())) for n in item.findall(".//parametername")]
        nm = ", ".join(f"`{n}`" for n in names if n)
        desc = _ws(render_desc(item.find("parameterdescription")))
        lines.append(f"- {nm} — {desc}" if desc else f"- {nm}")
    return "\n".join(lines)


def render_simplesect(ss) -> str:
    label = {
        "return": "Returns", "note": "Note", "warning": "Warning",
        "see": "See also", "attention": "Attention", "remark": "Remark",
        "since": "Since",
    }.get(ss.get("kind"))
    inner = _ws(render_desc(ss))
    return f"**{label}:** {inner}" if label else inner


def render_list(lst, ordered=False) -> str:
    out = []
    for i, item in enumerate(lst.findall("listitem"), 1):
        prefix = f"{i}." if ordered else "-"
        out.append(f"{prefix} {_ws(render_desc(item))}")
    return "\n".join(out)


def render_code(pl) -> str:
    body = "\n".join("".join(cl.itertext()) for cl in pl.findall("codeline"))
    return f"```\n{body}\n```"


def render_para(node) -> str:
    out, buf = [], []

    def flush():
        if buf:
            txt = _ws("".join(buf))
            if txt:
                out.append(txt)
            buf.clear()

    if node.text:
        buf.append(node.text)
    for ch in node:
        tag = ch.tag
        if tag == "parameterlist":
            flush(); out.append(render_paramlist(ch))
        elif tag == "simplesect":
            flush(); out.append(render_simplesect(ch))
        elif tag in ("itemizedlist", "orderedlist"):
            flush(); out.append(render_list(ch, ordered=(tag == "orderedlist")))
        elif tag == "programlisting":
            flush(); out.append(render_code(ch))
        else:
            buf.append(render_inline(ch))
        if ch.tail:
            buf.append(ch.tail)
    flush()
    return "\n\n".join(o for o in out if o.strip())


def render_desc(elem) -> str:
    """Render a Doxygen <briefdescription>/<detaileddescription> to markdown."""
    if elem is None:
        return ""
    return "\n\n".join(
        b for b in (render_para(p) for p in elem.findall("para")) if b.strip()
    )


def brief(m) -> str:
    return _ws(render_desc(m.find("briefdescription")))


def signature(m) -> str:
    defn = (m.findtext("definition") or "").strip()
    args = (m.findtext("argsstring") or "").strip()
    return _ws(defn + args)


_PY_SECTIONS = {"Parameters", "Returns", "Yields", "Examples", "Example",
                "Notes", "Note", "Raises", "Attributes", "See Also", "References"}


def py_summary(obj, name: str = "") -> str:
    """First paragraph of a Python docstring (stops at the first NumPy section).

    Reads ``__doc__`` directly (not ``inspect.getdoc``, which would pull in
    inherited/overload text) and strips pybind11's auto-generated signature
    preamble so we keep prose, not ``name(self: ...) -> ret`` noise.
    """
    if obj is None:
        return ""
    doc = obj.__doc__
    if not doc or not doc.strip():
        return ""
    out = []
    for ln in doc.splitlines():
        s = ln.strip()
        if not s:
            if out:
                break
            continue
        # Skip pybind signature lines / overload scaffolding.
        if name and s.startswith(name + "("):
            continue
        if not out and re.match(r"^\w+\(.*\)\s*(->.*)?$", s):
            continue
        if s == "Overloaded function." or re.match(r"^\d+\.\s+\w+\(", s):
            continue
        if ("self:" in s and "->" in s) or "phasic_pybind" in s:
            continue
        if s in _PY_SECTIONS:
            break
        out.append(s)
    return _ws(" ".join(out))


def cell(s: str) -> str:
    return s.replace("|", "\\|")


def make_anchor(page: str, name: str, seen: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_.]", "_", f"cpp.{page}.{name}")
    a, i = base, 2
    while a in seen:
        a, i = f"{base}_{i}", i + 1
    seen.add(a)
    return a


# --------------------------------------------------------------------------- #
# Loading Doxygen compounds
# --------------------------------------------------------------------------- #
def load_index():
    """simple class/struct name -> refid, for phasic:: compounds."""
    root = ET.parse(XML / "index.xml").getroot()
    out = {}
    for c in root.findall("compound"):
        if c.get("kind") in ("class", "struct"):
            name = c.findtext("name") or ""
            if name.startswith("phasic::"):
                out[name.split("::")[-1]] = c.get("refid")
    return out


def compounddef(refid):
    return ET.parse(XML / f"{refid}.xml").find(".//compounddef")


# --------------------------------------------------------------------------- #
# Emit: C++ class page
# --------------------------------------------------------------------------- #
def class_member_body(m, is_graph: bool):
    """(summary_for_table, full_body_markdown) for one C++ method."""
    name = m.findtext("name")
    pyobj, pyname = None, None
    if is_graph:
        pyname = RENAME.get(name, name)
        pyobj = getattr(phasic.Graph, pyname, None)

    b = brief(m)
    summary = b or py_summary(pyobj, pyname or "")
    detailed = render_desc(m.find("detaileddescription"))

    parts = []
    if summary:
        parts.append(summary)
    if detailed:
        parts.append(detailed)
    if pyobj is not None:
        parts.append(
            f"*Python equivalent:* "
            f"[`Graph.{pyname}`](../api/Graph.qmd#phasic.Graph.{pyname})"
        )
    return summary, "\n\n".join(p for p in parts if p.strip())


def emit_class(simple: str, refid: str) -> None:
    cd = compounddef(refid)
    is_graph = simple == PY_DOC_CLASS
    py_class = getattr(phasic, simple, None)

    funcs, statics, attribs = [], [], []
    for sd in cd.findall("sectiondef"):
        kind = sd.get("kind")
        if kind == "public-func":
            funcs += [m for m in sd.findall("memberdef")
                      if not (m.findtext("name") or "").startswith("~")]
        elif kind == "public-static-func":
            statics += list(sd.findall("memberdef"))
        elif kind == "public-attrib":
            for m in sd.findall("memberdef"):
                t = flatten_text(m.find("type"))
                if "*" not in t and "ptd_" not in t:  # drop raw C handles
                    attribs.append(m)

    methods = funcs + statics
    cls_desc = _ws(render_desc(cd.find("briefdescription")))
    if not cls_desc:
        cls_desc = py_summary(py_class)

    lines = [f"# phasic::{simple} {{ #cpp.{simple} }}", ""]
    if cls_desc:
        lines += [cls_desc, ""]

    seen: set = set()
    details = []          # (name, anchor, signature, body)
    for m in methods:
        name = m.findtext("name")
        anchor = make_anchor(simple, name, seen)
        summary, body = class_member_body(m, is_graph)
        details.append((name, anchor, signature(m), body, summary))

    if methods:
        lines += ["## Methods", "", "| Name | Description |", "| --- | --- |"]
        done = set()
        for name, anchor, _sig, _body, summary in details:
            if name in done:
                continue
            done.add(name)
            lines.append(f"| [{name}](#{anchor}) | {cell(summary)} |")
        lines.append("")
        for name, anchor, sig, body, _summary in details:
            lines += [f"### {name} {{ #{anchor} }}", "",
                      "```cpp", sig, "```", ""]
            if body:
                lines += [body, ""]

    if attribs:
        lines += ["## Attributes", "", "| Name | Type | Description |",
                  "| --- | --- | --- |"]
        for m in attribs:
            lines.append(
                f"| `{m.findtext('name')}` | `{cell(flatten_text(m.find('type')))}` "
                f"| {cell(brief(m))} |"
            )
        lines.append("")

    (OUT / f"{simple}.qmd").write_text("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------- #
# Emit: C API category pages
# --------------------------------------------------------------------------- #
def categorize(name: str) -> str:
    for slug, _title, _desc, subs in C_CATEGORIES:
        if any(s in name for s in subs):
            return slug
    return C_MISC[0]


def load_c_functions():
    cd = compounddef("phasic_8h")
    funcs = []
    for sd in cd.findall("sectiondef"):
        if sd.get("kind") == "func":
            funcs += list(sd.findall("memberdef"))
    return funcs


def emit_c_category(slug, title, desc, members) -> None:
    lines = [f"# {title} {{ #cpp.{slug} }}", ""]
    if desc:
        lines += [desc, ""]

    seen: set = set()
    details = []
    for m in members:
        name = m.findtext("name")
        anchor = make_anchor(slug, name, seen)
        details.append((name, anchor, signature(m), brief(m),
                        render_desc(m.find("detaileddescription"))))

    lines += ["## Functions", "", "| Name | Description |", "| --- | --- |"]
    for name, anchor, _sig, summary, _body in details:
        lines.append(f"| [{name}](#{anchor}) | {cell(summary)} |")
    lines.append("")
    for name, anchor, sig, summary, body in details:
        lines += [f"### {name} {{ #{anchor} }}", "", "```c", sig, "```", ""]
        full = "\n\n".join(p for p in (summary, body) if p.strip())
        if full:
            lines += [full, ""]

    (OUT / f"{slug}.qmd").write_text("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------- #
# Emit: index + sidebar
# --------------------------------------------------------------------------- #
def emit_index(class_rows, c_pages) -> None:
    lines = [
        "# C / C++ API Reference {.doc .doc-index}",
        "",
        "The C++ classes (`api/cpp/phasiccpp.h`, `api/cpp/scc_graph.h`) and the C "
        "core (`api/c/phasic.h`) that the [Python API](../api/index.qmd) is built "
        "on. The Python `Graph` *is* the C++ `phasic::Graph`; where a C++ method "
        "has a Python counterpart it links across to the Python reference.",
        "",
        "## C++ Classes",
        "",
        "| | |",
        "| --- | --- |",
    ]
    for simple, summary in class_rows:
        lines.append(f"| [phasic::{simple}]({simple}.qmd#cpp.{simple}) | {cell(summary)} |")
    lines += ["", "## C API", "", "| | |", "| --- | --- |"]
    for slug, title, desc, n in c_pages:
        lines.append(f"| [{title}]({slug}.qmd#cpp.{slug}) | {cell(desc)} ({n} functions) |")
    (OUT / "index.qmd").write_text("\n".join(lines).rstrip() + "\n")


def emit_sidebar(class_names, c_pages) -> None:
    lines = ["website:", "  sidebar:", "  - id: cpp_api", "    contents:",
             "    - cpp_api/index.qmd",
             "    - section: C++ Classes", "      contents:"]
    lines += [f"      - cpp_api/{n}.qmd" for n in class_names]
    lines += ["    - section: C API", "      contents:"]
    lines += [f"      - cpp_api/{slug}.qmd" for slug, *_ in c_pages]
    (OUT / "_sidebar.yml").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    if not XML.exists():
        sys.exit(f"Doxygen XML not found at {XML}. Run `doxygen Doxyfile` in docs/ first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT.glob("*.qmd")) + list(OUT.glob("_sidebar.yml")):
        stale.unlink()

    index = load_index()

    # C++ classes
    class_rows = []
    for simple in CPP_CLASS_ORDER:
        refid = index.get(simple)
        if refid is None:
            sys.exit(f"C++ class {simple} not found in Doxygen index.")
        emit_class(simple, refid)
        py_class = getattr(phasic, simple, None)
        cd = compounddef(refid)
        summary = _ws(render_desc(cd.find("briefdescription"))) or py_summary(py_class)
        class_rows.append((simple, summary))

    # C API
    funcs = load_c_functions()
    by_cat = {}
    for m in funcs:
        by_cat.setdefault(categorize(m.findtext("name")), []).append(m)

    c_pages = []
    for slug, title, desc, _subs in C_CATEGORIES:
        members = by_cat.get(slug)
        if members:
            emit_c_category(slug, title, desc, members)
            c_pages.append((slug, title, desc, len(members)))
    if by_cat.get(C_MISC[0]):
        slug, title, desc = C_MISC
        members = by_cat[slug]
        emit_c_category(slug, title, desc, members)
        c_pages.append((slug, title, desc, len(members)))

    emit_index(class_rows, c_pages)
    emit_sidebar(CPP_CLASS_ORDER, c_pages)

    n_c = sum(n for *_x, n in c_pages)
    print(f"[gen_cpp_api] {len(CPP_CLASS_ORDER)} C++ classes, "
          f"{n_c} C functions across {len(c_pages)} category pages -> {OUT}")
    for slug, title, _desc, n in c_pages:
        print(f"    {slug:18} {n:3}  {title}")


if __name__ == "__main__":
    main()
