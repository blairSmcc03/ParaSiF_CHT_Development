# ParaSiF_CHT_Development

This repository contains the ongoing development of ParaSiF new feature on **Conjugate Heat Transfer** between FEniCSx and OpenFOAM that to be integrated into the ParaSiF Parallel Partitioned Simulation Framework once mature.
It is maintained as a **submodule** of the main ParaSiF repository: [ParaSiF Main Repository](https://github.com/ParaSiF/ParaSiF). Please note that all dev-submodules will not be shown in the main ParaSiF repository for users.

---

## Overview

Ongoing development of ParaSiF new feature on **Conjugate Heat Transfer** between FEniCSx and OpenFOAM via the **[MUI coupling library](https://mxui.github.io/)**.

Key features to be developed:

- Thermal conduction in solid domain
- Conjugate Heat Transfer between fluid and structure domain
- Conjugate Heat Transfer between two solid domains
- Thermal expension in solid domain

---

## Compatible Codebase

This solver has been tested and is compatible with **[FEniCSx v0.9.0](https://github.com/FEniCS/dolfinx/releases/tag/v0.9.0.post1)** and **[OpenFOAM v2506](https://www.openfoam.com/news/main-news/openfoam-v2506)**.

> Users are recommended to use this version to ensure full compatibility with ParaSiF FEniCSx solvers.

---

## (Suggested) Location in the Main ParaSiF Repository

`ParaSiF/dev/CHT/`

---

## Repository Structure

```
ParaSiF/dev/CHT/
├── fluid/                # fluid OpenFOAM solvers folder
│ ├── src/                # ParaSiF-specific OpenFOAM source code folder
│ │ ├── solvers/          # ParaSiF-specific OpenFOAM solvers
│ │ └── libs/             # Libs used by OpenFOAM solvers
│ └── test/               # OpenFOAM unit test folder
├── structure/            # structure FEniCSx solvers folder
│ ├── src/                # ParaSiF-specific FEniCSx source code folder
│ └── test/               # FEniCSx unit test folder
└── example/              # example and integrated test folder
```

---

## Installation

**Note:** This new feature will be a part of ParaSiF. Follow the main ParaSiF repository instructions to initialise submodules and install global dependencies.

### Steps

1. **Obtain and install the codebase**
   - Initialise the FEniCSx and OpenFOAM submodules from the main ParaSiF repository.
   - Ensure the FEniCSx and OpenFOAM submodules in the main ParaSiF repository are correctly installed by following their instructions.

2. **Install the OpenFOAM solver in this repository**
   **To Be developed**

## Running Tests and Example Cases

Benchmark cases are located in the test/ folder:

### Steps

1. Navigate to the desired benchmark folder:

```bash
cd XXX/test/XXX
```

2. Run the simulation:

```bash
./Allrun.sh
```

3. (Optional) Clean up previous results before rerunning:

```bash
./Allclean.sh
```
4. Check results:

## Contributing

ParaSiF, including this repository, is an **open-source project**, and contributions from the community are warmly welcomed.

There are many ways you can help improve this submodule, including:

- Adding new features, libs or solvers
- Improving documentation, tests and examples
- Fixing bugs or refining existing functionality
- Sharing feedback and suggestions for enhancements

Your contributions, whether large or small, are highly valued and help make ParaSiF a stronger resource for the research community.

For detailed guidance on contributing, please see the [CONTRIBUTING.md](https://github.com/ParaSiF/ParaSiF/blob/main/CONTRIBUTING.md) in the main ParaSiF repository.

## License

Copyright (C) 2021–2025 The ParaSiF Development Team.  
Licensed under the **GNU General Public License v3 (GPL-3.0)**.

## Contact

For questions or contributions, please contact the ParaSiF Development Team

