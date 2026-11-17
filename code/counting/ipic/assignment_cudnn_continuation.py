"""Apply the validated, equally matched runtime setting to the frozen runner."""
import hashlib
import json
import pathlib
import sys

import torch

from . import assignment_continuation


def main():
    # All math, batches, authentic states, assignments and endpoints stay in the
    # unchanged runner. This setting applies to both serial comparison arms.
    torch.backends.cudnn.deterministic = True
    folder = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
    record = {'cudnn_deterministic': True, 'full_cuda_bitwise_determinism_claim': False,
              'same_runtime_both_arms': True, 'numerical_tolerances_unchanged': True,
              'preserved_runner_sha256': hashlib.sha256(pathlib.Path(assignment_continuation.__file__).read_bytes()).hexdigest(),
              'amendment': 'Deterministic cuDNN algorithm selection after measured unchanged-control backward nondeterminism; no model/loss/data/optimizer/budget change.'}
    with (folder / 'runtime_amendment.json').open('x') as f: json.dump(record, f, indent=2)
    assignment_continuation.main()


if __name__ == '__main__': main()
