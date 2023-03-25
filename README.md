# ***YOUR PROJECT NAME HERE***



<!-- [              MARKDOWN BADGES              ] -->

|||
|---|---|
|**Research**| [![Paper](http://img.shields.io/badge/paper-arXiv.0000.0000-B31B1B.svg)](https://www.arXiv.org/abs/0000.0000) [![Conference](http://img.shields.io/badge/conference-year-4b44ce.svg)]() |
|**Environment**| [![Python](https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/) [![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)](https://github.com/Lightning-AI/lightning) [![CUDA](https://img.shields.io/badge/CUDA-11.7-76B900.svg?logo=nvidia&logoColor=white)]() |
|**Metadata**| [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |

<!-- [                    END                    ] -->



<!-- [             TABLE OF CONTENTS             ] -->

<details>
<summary><strong>Table of Contents</strong></summary>

- [:wave: Overview](#wave-overview)
- [:gear: Prerequisites](#gear-prerequisites)
- [:file_folder: Directories](#file_folder-directories)  
    - [:sparkles: project/](#sparkles-project)
    - [:test_tube: tests/](#test_tube-tests)
    - [:desktop_computer: scripts/](#desktop_computer-scripts)
    - [:ledger: notebooks/](#ledger-notebooks)
    - [:floppy_disk: saves/](#floppy_disk-saves)
- [:mag: References](#mag-references)  
- [:scroll: Citation](#scroll-citation)
- [:balance_scale: License](#balance_scale-license)  

</details>

<!-- [                    END                    ] -->



## :wave: **Overview**

*YOUR OVERVIEW HERE*

## :gear: **Prerequisites**

First, clone this repository:

```bash
$ git clone https://www.github.com/YOUR/REPOSITORY project
$ cd project
```

Next, install the [dependencies] via pip:

```bash
$ pip install [-U] -r requirements.txt

# Using a virtual environment is recommended
$ python -m venv .venv
$ .venv/Scripts/activate
(.venv) $ pip install [-U] -r requirements.txt
```

If you want to modify or test the source codes, install the [dev-dependencies]:

```bash
$ pip install [-U] -r requirements-dev.txt
```

[dependencies]: ./requirements.txt
[dev-dependencies]: ./requirements-dev.txt

## :file_folder: **Directories**

### :sparkles: `project/`

> :fire: **REMOVE THIS BLOCK QUOTE IN YOUR PROJECT README**  
> If you rename this folder, you should rename `project/` and change links as well.

[project/] *self-contains* all **main** source codes, and consists of submodules such as [models/] and [datasets/].

To use these source codes inside your project, copy this folder inside your project, and see below:

```python
# Import `MyModel` from `project/models/path/to/file.py`
from project.models.path.to.file import MyModel

# Import `MyDataset` from `project/datasets/path/to/file.py`
from project.datasets.path.to.file import MyDataset
```

[project/]: ./project/
[models/]: ./project/models
[datasets/]: ./project/datasets/

### :test_tube: `tests/`

[tests/] contains every test case of the source codes of this project.  
To run them, you should install the [pytest][pytest-url] *included in the [dev-dependencies]*.  
> [This extension][extension-test-url] may help you to run the test cases conveniently.

[tests/]: ./tests/
[pytest-url]: https://docs.pytest.org/en/7.2.x/
[extension-test-url]: https://marketplace.visualstudio.com/items?itemName=LittleFoxTeam.vscode-python-test-adapter

### :desktop_computer: `scripts/`

[scripts/] contains *self-runnable* scripts for  training, validation, testing, visualization, etc.  

In most cases, you can run the scripts by choosing one of these two commands:

```bash
$ python ./scripts/SCRIPT_NAME.py [OPTION ... [--FLAG=VALUE ...]]

$ python -m scripts.SCRIPT_NAME [OPTION ... [--FLAG=VALUE ...]]
```

Some scripts may show you how to use them:

```bash
$ python ./scripts/SCRIPT_NAME.py --help

# the script may print its manual ...
```

[scripts/]: ./scripts/

### :ledger: `notebooks/`

[notebooks/] contains [jupyter notebook][jupyter-url] files for training, validation, testing, visualization, etc.  
> [This extension][extension-jupyter-url] may help you to run the notebooks conveniently.

[notebooks/]: ./notebooks/
[jupyter-url]: https://jupyter.org/
[extension-jupyter-url]: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter

### :floppy_disk: `saves/`

[saves/] contains static files such as checkpoints, logs, and prediction results.  

[saves/]: ./saves/

## :mag: **References**

*YOUR REFERENCES HERE*

## :scroll: **Citation**

*YOUR CITATION HERE*

## :balance_scale: **License**

This project is distributed under the terms of [*YOUR LICENSE HERE*](./LICENSE) license.  

<!-- (   REMOVE THIS FROM YOUR PROJECT README    ) -->

> :fire: **REMOVE THIS BLOCK QUOTE FROM YOUR PROJECT README**    
> The [lightning-project-template] is under the [MIT] license.  
> Change the [LICENSE](./LICENSE) file and [license-badge] for your project.
>> Even if your project is under the same license as the template,  
>> `Copyright (c) 2023 Jaewoo Park` (line 3) must be modified.

[lightning-project-template]: https://github.com/kaparoo/lightning-project-template
[MIT]: https://opensource.org/licenses/MIT
[license-badge]: https://gist.github.com/lukas-h/2a5d00690736b4c3a7ba

<!-- (                    END                    ) -->