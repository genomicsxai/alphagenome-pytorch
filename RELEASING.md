# Releasing alphagenome-pytorch

## Prerequisites

- Push access to the repository

PyPI trusted publishing has already been configured for this repository.

## Release Process

1. **Update CHANGELOG**

2. **Create and push a version tag:**
   ```bash
   git tag v0.4.0
   git push origin v0.4.0
   ```

3. **Monitor the release:**
   - Go to Actions tab → "Publish to PyPI" workflow
   - Verify build and publish steps complete

4. **Verify on PyPI:**
   - https://pypi.org/project/alphagenome-pytorch/

## Version Format

Versions are derived from git tags via `hatch-vcs`:
- Tagged commit `v0.5.0` → version `0.5.0`
- 3 commits after tag → version `0.5.0.dev3+g1a2b3c4`

## Manual Release (if needed)

```bash
# Build
hatch build

# Upload (requires API token)
hatch publish
```

## First-Time PyPI Setup

1. Create project on PyPI (first publish) or ensure you have owner access
2. Go to PyPI → Project → Settings → Publishing
3. Add trusted publisher:
   - **Owner**: `genomicsxai`
   - **Repository**: `alphagenome-pytorch`
   - **Workflow**: `publish.yml`
   - **Environment**: `pypi`

## GitHub Environment Setup (Optional but recommended)

1. Go to repo Settings → Environments
2. Create environment named `pypi`
3. Optionally add protection rules
