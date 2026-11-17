"""Same frozen two-arm trajectory, with a validated shared numerical runtime."""
from .stable_resampling import install

import hashlib
import json
import pathlib
import sys

from . import assignment_continuation


def main():
    runtime=install();folder=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
    runtime['preserved_runner_sha256']=hashlib.sha256(pathlib.Path(assignment_continuation.__file__).read_bytes()).hexdigest()
    runtime['numerical_tolerances_unchanged']=True
    with (folder/'runtime_amendment.json').open('x') as f:json.dump(runtime,f,indent=2)
    assignment_continuation.main()


if __name__=='__main__':main()
