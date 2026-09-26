"""Helpers for TerraSR_Colab.ipynb.

Imported by the notebook cells as:

    sys.path.insert(0, str(REPO_DIR / "colab"))
    from colab_utils import stream_run
"""
import os
import subprocess
import sys
import time


def stream_run(cmd, cwd=None, label=None, check=False):
    """Run a command, printing its output line by line *as it happens*.

    subprocess.run() looks like it streams, but the child's stdout is a pipe,
    not a terminal, so Python line-buffers it only when it thinks it is
    interactive — in a notebook it uses a block buffer instead. A long training
    run therefore prints nothing for hours and then dumps every epoch at once
    when the process exits, which is useless in Colab where the session can be
    cut at any moment. PYTHONUNBUFFERED + `-u` + reading the pipe line by line
    makes progress appear while it is happening.

    Returns the child's exit code (or raises RuntimeError when check=True).
    """
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] == sys.executable and "-u" not in cmd:
        cmd.insert(1, "-u")                      # unbuffered child interpreter
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    try:
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\ninterrupted. Checkpoints for every completed epoch are intact — "
              "re-run this cell to resume.", flush=True)
        raise
    finally:
        proc.stdout.close()

    mins = (time.time() - t0) / 60
    tag = label or cmd[0]
    if code == 0:
        print(f"\n{tag} finished OK ({mins:.1f} min)", flush=True)
    else:
        print(f"\n{tag} exited with code {code} after {mins:.1f} min", flush=True)
        if check:
            raise RuntimeError(f"{tag} failed (exit {code}) - see output above")
    return code
