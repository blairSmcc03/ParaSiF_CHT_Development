# Unit Test on FEniCSx Solid Thermal Solver -- Two-Way Coupling with a Dummy Fluid C++ Code via MUI

This unit test demonstrates and verifies a two-way coupling of:

- A transient solid thermal solver implemented in FEniCSx,
- A dummy fluid code written in C++,
- coupling is achieved by using the MUI library.

---

## Case Overview

### Domain schematic

```
                          MUI             +z direction
                       interface      (perpendicular inward)
                           |                   ↑
                           |                   │
        ┌──────────────┐   |   ┌──────────────────────────────┐
        │              │   |   │                              │
        │ Fluid Domain │   |   │        Solid Domain Ω        │
        │              │   |   │                              │
        │              │   |   │                              │<-- y direction
        │              │   |   │                              │
        ├──────────────┤   |   ├──────────────────────────────┤
       x = -1.5    x = -1  |  x = -1 (Left boundary)         x = +1 (Right boundary)
           Neumann BC──┤   |   └──── Dirichlet BC             └── Dirichlet BC (T = 0k)
       Coupled via MUI ↑   |   ↑   Coupled via MUI
                       │   |   │
          receives q_flux  |  receives T
             sends T=500k  |  sends q_flux
                           |
```

### FEniCSx transient solid thermal solver

- Simulates transient heat conduction in a 3-D cuboid domain: [-1, 1] X [-2, 2] X [-1, 1]. 
- Employs an implicit backward-Euler time integrator.
- Left boundary (x = −1.0): Dirichlet temperature boundary (reveive temperature from the fluid domain via MUI).
- Right boundary (x = +1.0): Dirichlet temperature boundary (T = 0k).
- During the time loop, the left boundary temperature evolves through MUI-fetched values from the fluid doamin, while the solid pushes computed heat flux of the left boundary back to the fluid domain.

### C++ dummy fluid script

- Represents a placeholder for a real CFD solver in a 3-D cuboid domain: [-1.5, -1] X [-2, 2] X [-1, 1].
- Exchanges boundary data with FEniCSx via MUI:
- Fetches heat-flux components (heatFluxx, heatFluxy, heatFluxz) at the coupling interface from the solid domain.
- Pushes temperature values at the coupling interface to the solid.
- All values are fixed or synthetic for testing MUI coupling logic.

---

## Coupling Configuration
| **Exchange Direction** | **Field Name**                   | **Quantity** | **Description**                                      |
|-------------------------|----------------------------------|--------------|------------------------------------------------------|
| Fluid → Solid           | `temperature`                   | Scalar       | Dirichlet boundary temperature imposed on the solid  |
| Solid → Fluid           | `heatFluxx`, `heatFluxy`, `heatFluxz` | Vector       | Neumann boundary heat flux vector imposed on the coupling interface of fluid (fixed gradient temperature)      |


---

## Compatible Codebase

This solver has been tested and is compatible with **[FEniCSx v0.9.0](https://github.com/FEniCS/dolfinx/releases/tag/v0.9.0.post1).

> Users are recommended to use this version to ensure full compatibility with ParaSiF FEniCSx solvers.

---

## Execution

1. Run the simulation:

```bash
./Allrun.sh
```

2. (Optional) Clean up previous results before rerunning:

```bash
./Allclean.sh
```
3. Check results:

At the end of the simulation, the Python script `compareSolution.py` in `structureDomain/` folder will be executed to generate comparison plots `compare_t*.png` in `structureDomain/`. Solver output file `fenicsx_solid_coupled.xdmf` is also availible in `structureDomain/` for further analysis in Paraview.

---

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

