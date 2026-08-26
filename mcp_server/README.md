# Lacuna MCP server

Exposes Lacuna's cryptic-pocket discovery to Claude Desktop and Claude Code as a
local tool. Claude launches it as a subprocess over stdio, it runs Lacuna on your
own machine, and nothing leaves it: no hosting, no API key, no account.

Lacuna's default backend is CPU normal mode analysis and finishes in seconds, so
this is reasonable to run interactively during a conversation.

## Install

```bash
pip install lacuna-pockets mcp
```

Then get this server. It is not part of the pip package, so either clone the
repository:

```bash
git clone https://github.com/mooreneural/lacuna.git
```

or download the single file it needs:

```bash
curl -O https://raw.githubusercontent.com/mooreneural/lacuna/main/mcp_server/lacuna_mcp.py
```

Updating means pulling or re-downloading this file. The science updates
separately and does not need that: the server calls whichever `lacuna` is
installed, so `pip install -U lacuna-pockets` is picked up on the next call.

## Register

**Claude Code**

```bash
claude mcp add lacuna -- python /absolute/path/to/mcp_server/lacuna_mcp.py
```

**Claude Desktop**, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lacuna": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/lacuna_mcp.py"]
    }
  }
}
```

The path must be absolute. If `lacuna` lives in a virtualenv, point `command` at
that environment's `python` rather than a bare `python`.

## Tools

### `find_cryptic_pockets`

Ranked cryptic sites for a structure. Takes a local `.pdb`/`.cif` path or a
four-character PDB id, which is fetched from RCSB and cached. Each site returns a
centroid, lining residues, volume across the ensemble and in the starting
structure, persistence, crypticity, and druggability.

Options: `conformers` (default 20), `top`, `min_crypticity`,
`min_druggability`, and `rank_by` (`learned`, or `learned-plm` which ranks better
but downloads a 2.5 GB model on first use).

### `export_docking_region`

Runs the same discovery and returns docking inputs: AutoDock Vina and GNINA
search boxes, or Boltz constraint files.

## Worth knowing

**These are search regions, not receptors.** A cryptic site is closed in the
deposited structure, and a generated conformer opens it only partly. Rigid
docking against either file will fight the receptor. Use the region to focus
flexible-receptor docking, or as a co-folding constraint.

**Ranked sites are candidates, not validated binding sites.** On the CryptoBench
test fold Lacuna's top five contain the annotated site for about 66% of targets.
Treat the ranking as a shortlist.

**Crypticity above roughly 0.3** indicates a site that opens substantially
relative to the starting structure. Lower values are sites already largely open,
where a single-structure detector such as fpocket or P2Rank is a better fit.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LACUNA_MCP_CACHE` | system temp | Where fetched PDB files are cached |
| `LACUNA_MCP_TIMEOUT` | 900 | Seconds before a run is abandoned |

## Implementation note

The tools shell out to the installed `lacuna` CLI rather than importing the
pipeline. That keeps the server working against whatever version is installed,
isolates a crash in the science code from the transport, and means the code path
that runs is the tested one.
