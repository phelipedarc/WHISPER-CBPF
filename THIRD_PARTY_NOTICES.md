# Third-party notices

WHISPER-CBPF is distributed under the GNU General Public License v3 (see `LICENSE`). Part of its
model library is adapted from two GPL-3.0 projects by Nikhil Sarin and collaborators, and part from
MOSFiT (MIT licence). This file says what was taken, from where, which files carry it, and which
papers the models implement.

## The projects

**redback** (Sarin et al.), <https://github.com/nikhil-sarin/redback>, GPL-3.0. A Python package of
electromagnetic transient models and inference. WHISPER-CBPF reads redback's priors and models at
run time (the `[models]` extra) and also contains JAX re-implementations of some of its models,
written against redback 1.20. Citation, as given in the adapted files: Sarin, N. et al. 2024,
MNRAS 531, 1203 (<https://arxiv.org/abs/2308.12806>).

**redback-jax** (Nikhil Sarin et al.), <https://github.com/nikhil-sarin/redback-jax>, GPL-3.0,
commit 3431e2e. A JAX companion to redback. The files in the first table below are adapted from it
and from redback.

**MOSFiT** (James Guillochon et al.), <https://github.com/guillochon/MOSFiT>, MIT licence, commit
0bc772e (version 2.1.1), Copyright (c) 2016-2026 James Guillochon. The Modular Open Source Fitter
for Transients; its `tde` model is adapted in the files of the table "Adapted from MOSFiT" below.
Its licence is reproduced at the end of that section.

## What was adapted

Each file below begins with an attribution header. Where the original differed from redback's own
result, the code here was corrected to agree with redback; the arithmetic that was rewritten for
traceability, stability or speed is described in each file's module docstring, and its agreement with
redback is measured in `tests/test_adopted_phenomenological.py`, `tests/test_adopted_tde.py`,
`tests/test_adopted_kilonova.py` and `tests/test_adopted_supernova.py`.

### Adapted from redback-jax and from redback

| File | Model |
|---|---|
| `whisper_cbpf/models/jax/phenomenological.py` | smooth exponential power law, exponential rise with power-law decline, Bazin, Villar |
| `whisper_cbpf/models/jax/_phenomenological_spec.py` | parameter names, per-band layout and priors of the four phenomenological models |
| `whisper_cbpf/models/jax/tde_fallback.py` | `tde_fallback`: fall-back of a disrupted star with viscous delay (its fall-back engine is MOSFiT's, see "Adapted from MOSFiT") |
| `whisper_cbpf/models/jax/kilonova_metzger.py` | `metzger_kilonova_model` and `metzger_magnetar_driven_kilonova_model` |
| `whisper_cbpf/models/jax/csm_interaction.py` | `csm_interaction`: Chevalier-Chatzopoulos CSM interaction |
| `whisper_cbpf/models/jax/shock_models.py` | `shocked_cocoon` |
| `whisper_cbpf/models/jax/magnetar_ode.py` | `general_magnetar_driven_supernova` |

### Adapted from redback only

| File | Model |
|---|---|
| `whisper_cbpf/models/jax/tde_analytical.py` | `tde_analytical`: a `t^-5/3` fall-back through Arnett diffusion. redback-jax implements this model as well; no code is taken from it. |
| `whisper_cbpf/models/jax/tde_stream_stream.py` | `stream_stream_tde`: stream-stream collisions with a Gaussian rise (the engine, `tde.stream_stream_collision`, is redback's `_stream_stream_collision`; redback-jax has no counterpart, nothing is taken from it). |

### Adapted from MOSFiT

| File | Model |
|---|---|
| `whisper_cbpf/models/jax/tde_mosfit.py` | `tde` of MOSFiT 2.1.1 (James Guillochon et al., <https://github.com/guillochon/MOSFiT>, commit 0bc772e, MIT licence, Copyright (c) 2016-2026 James Guillochon): the `TdePhotosphere`, `TDEConstraints` and `Kroupa` modules. Its fall-back engine and viscous kernel are those of `tde_fallback.py` (imported). The Guillochon & Ramirez-Ruiz tables MOSFiT ships are byte-identical to the ones redback distributes. |
| `whisper_cbpf/models/jax/tde_fallback.py` | The fall-back engine of `tde_fallback` and `tde_mosfit`: MOSFiT 2.1.1's `Fallback` (`mosfit/modules/engines/fallback.py`; redback's `_initialize_mosfit_tde_model` and `_tde_mosfit_engine` are the same code). `Fallback.__init__` is `build_fallback_tables` (run once; its output is `whisper_cbpf/data/tde_fallback_nodes.npz`) and `Fallback.process` is `fallback_engine`. The file is also adapted from redback-jax and redback (above). |

MOSFiT's licence, reproduced from its `LICENSE` file:

> MIT License
>
> Copyright (c) 2016-2026 James Guillochon
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Other files

`whisper_cbpf/models/jax/supernova.py` also contains one adapted engine, `shock_cooling` (the Piro
engine bound on its own, with its own photosphere), and the table that lists the four adopted
supernova-family models; the file's header says so. The adapter shared by the two TDE models
(`whisper_cbpf/models/jax/_adopted_tde.py`) and the redback transcription of the `tde_analytical`
constraints (`whisper_cbpf/models/constraints.py`) are written for WHISPER-CBPF against redback's
interface and behaviour.

### Data tables

| File | Provenance |
|---|---|
| `whisper_cbpf/data/csm_table.txt` | The table of shock coefficients redback ships as `redback/tables/csm_table.txt` (the same file, byte for byte), read by `csm_interaction`. |
| `whisper_cbpf/data/tde_fallback_nodes.npz` | MOSFiT's fall-back nodes, read by `tde_fallback` and `tde_mosfit`: the arrays MOSFiT's `Fallback.__init__` computes (each simulation's log time and log `dM/dt` split at the peak; per pair of neighbouring betas the mapped times, slopes and intercepts), padded to a fixed length, computed by `build_fallback_tables` in `tde_fallback.py` from the fall-back tables redback distributes in `redback/tables/guillochon_tde_data` (the `dM/dE` files for gamma = 4/3 and 5/3, which redback's `info.txt` says were converted from the `dmdts` of James Guillochon's blog; byte-identical to MOSFiT's `models/tde/data`). The file records the SHA-256 of every source file. |
| `whisper_cbpf/data/tde_fallback_tables.npz` | The tables of 0.3.0: the same source files resampled onto 512 equal steps of the normalised log time of each branch (by 0.3.0's `build_fallback_tables`). No code reads it since the engine became MOSFiT's node construction. |

## Called, not adapted: TiDE

**TiDE** (Zsofia V. Kovacs-Stermeczky and Jozsef Vinko), MIT license. `whisper_cbpf/models/tide.py` calls the installed C++
library `libtide` through its Python wrapper `tidepy`; no TiDE source, table or binary is included in or adapted by
WHISPER-CBPF (the fall-back tables `table_n3.dat` and `table_n3per2.dat` that libtide reads belong to its installation).
Citation: Kovacs-Stermeczky, Zs. V. & Vinko, J. 2023, PASP 135, 034102 (<https://arxiv.org/abs/2302.08441>). The model also
cites Lodato, King & Pringle 2009, MNRAS 392, 332; Strubbe & Quataert 2009, MNRAS 400, 2070; Lodato & Rossi 2011, MNRAS 410, 359.

TiDE's license, reproduced from its `LICENSE` file:

> MIT License
>
> Copyright (c) 2022 Kovács-Stermeczky Zsófia
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## The papers the models implement

As cited in the files named.

* **Phenomenological models** (`phenomenological.py`): Bazin et al. 2009
  (<https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract>); Villar et al. 2019
  (<https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract>); Arnett 1982
  (<https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract>). redback cites no reference for
  `smooth_exponential_powerlaw` or `exp_rise_powerlaw_decline`.
* **`tde_fallback`**: Mockler, Guillochon & Ramirez-Ruiz 2019, ApJ 872, 151; Guillochon &
  Ramirez-Ruiz 2013, ApJ 767, 25; Guillochon et al. 2018, ApJS 236, 6 (MOSFiT); Tout, Pols,
  Eggleton & Han 1996, MNRAS 281, 257 (the main-sequence radius: redback evaluates this relation
  and names its coefficients `Tout_*`); Sarin et al. 2024, MNRAS 531, 1203 (redback).
* **`tde_mosfit`**: Guillochon et al. 2018, ApJS 236, 6 (MOSFiT); Mockler, Guillochon & Ramirez-Ruiz 2019, ApJ 872, 151; Guillochon & Ramirez-Ruiz 2013, ApJ 767, 25; Kroupa 2001, MNRAS 322, 231 (the stellar-mass distribution); Tout et al. 1996, MNRAS 281, 257.
* **`tde_stream_stream`**: Piran, Svirski, Krolik, Cheng & Shiokawa 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Krolik, Piran & Ryu 2024, ApJ 973, 12 (the cooling-time factor); Sarin et al. 2024, MNRAS 531, 1203 (redback).
* **`tde_analytical`**: Arnett 1982, ApJ 253, 785; Villar et al. 2017, ApJ 851, L21; Nicholl,
  Guillochon & Berger 2017, ApJ 850, 55; Sarin et al. 2024, MNRAS 531, 1203 (redback).
* **Metzger kilonovae** (`kilonova_metzger.py`): Metzger 2017, Living Reviews in Relativity 20, 3;
  Barnes & Kasen 2016, ApJ 829, 110; Metzger et al. 2010, MNRAS 406, 2650; Tanaka et al. 2019
  (the electron fraction against opacity used by redback); Metzger & Piro 2014, MNRAS 439, 3916
  (magnetar-boosted kilonovae).
* **`csm_interaction`**: Chevalier 1982; Chatzopoulos et al. 2012, 2013; Villar et al. 2017;
  Jacobson-Galan et al. 2020.
* **`shocked_cocoon`**: Piro & Kollmeier 2018.
* **`shock_cooling`**: Piro 2021.
* **`general_magnetar_driven`**: Sarin et al. 2022; Omand & Sarin 2024; Nicholl et al. 2017;
  Villar et al. 2017.

## Using the code

The adapted files are GPL-3.0, as is the rest of the package; redistribution keeps their
attribution headers. The two files adapted from MOSFiT also carry MOSFiT's copyright and permission
notice in the file itself (the licence reproduced above), so that it travels with the code in a
wheel, whose metadata holds only `LICENSE`. If you use these models in published work, cite redback
(and redback-jax where its code is the source, MOSFiT for `tde_fallback` and `tde_mosfit`) as well
as WHISPER-CBPF; `CITATION.cff` lists the redback repositories and MOSFiT.
