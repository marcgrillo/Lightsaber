"""This file contains IO functionalities to perform benchmarking."""

import dataclasses
import pathlib
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.testing as npt

PathLike = Union[str, pathlib.Path]


def save_results_as_txt(results: Any, directory: PathLike,
                        filestem_to_result_field: Dict[str, str]):
  """Saves the results as txt files.

  Arguments:
    results: The dataclass that contains the results.
    directory: Folder to write the data into.
    filestem_to_result_field: A dictionary mapping filestems to field-names.
  """
  directory = pathlib.Path(directory)

  for filestem, field_name in filestem_to_result_field.items():
    data = getattr(results, field_name)
    np.savetxt(directory / f'{filestem}.csv', data, delimiter=' ')


def update_results_for_benchmarks(results: Any, directory: PathLike,
                                  mapping: Optional[dict[str, str]] = None):
  """Updates the benchmark data.

  Args:
    results: The dataclass that contains the results.
    directory: Folder to write the data into.
    mapping: Optional mapping from field name to file stem.
  """
  directory = pathlib.Path(directory)
  if directory.exists():
    for entry in directory.iterdir():
      entry.unlink()
  else:
    directory.mkdir(parents=True, exist_ok=True)

  if mapping is None:
    mapping = {}
  for field in dataclasses.fields(results):
    data = getattr(results, field.name)
    filename = mapping.get(field.name, field.name)
    np.save(directory / f'{filename}.npy', data)


def check_results_against_benchmarks(results: Any, directory: PathLike,
                                     mapping: Optional[dict[str, str]] = None):
  """Checks the results against the benchmark data.

  Args:
    results: The dataclass that contains the results.
    directory: Folder to write the data into.
    mapping: Optional mapping from field name to file stem.

  Raises:
    IsADirectoryError: Raised if the benchmark directory does not exist.
  """
  directory = pathlib.Path(directory)
  if not directory.exists():
    raise IsADirectoryError(
        f'Benchmark directory does not exist ("{directory}")'
        'Create benchmark data first( "--update_benchmarks").')
  if mapping is None:
    mapping = {}
  for field in dataclasses.fields(results):
    actual_data = getattr(results, field.name)
    filename = mapping.get(field.name, field.name)
    expected_data = np.load(directory / f'{filename}.npy')
    print(f'Checking {field.name} against benchmark ...')
    npt.assert_almost_equal(actual_data, expected_data, decimal=6)
  print('All fields are good.')
