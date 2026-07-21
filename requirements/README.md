# Reproducible environments

The `.in` files are maintained inputs. Their `.txt` counterparts are
hash-pinned sets generated with Python 3.11. The `.txt` extension is used
because `.lock` files are not stored in the OneDrive workspace.

Regenerate each file with `pip-compile --generate-hashes
--resolver=backtracking --strip-extras --no-emit-index-url
--no-emit-trusted-host`, and install with:

```powershell
python -m pip install --require-hashes -r requirements/runtime.txt
```

Regenerate and audit the sets on every dependency change. Add
Python-version-specific `.txt` sets if supported interpreters resolve
materially different graphs.
