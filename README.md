
# Genetic Network eXperts (GeNeX)

📄 **Paper (TNNLS, accepted):** *GeNeX: Genetic Network eXperts framework for addressing Validation Overfitting*

**GeNeX** is a machine learning framework for **robust classification under distribution shift** (when deployment data differs from training data).  
It **evolves diverse neural networks** and **fuses them into expert prototypes** that can be ensembled for improved robustness and reduced validation overfitting.



<img width="8945" height="3862" alt="Εικόνα3" src="https://github.com/user-attachments/assets/1f697345-84d3-4276-8a0d-1b9134a6f739" />


GenE generates a diverse pool M without validation monitoring via short supervised training and genetic crossover/mutation, encouraging broad weight-space exploration and limiting early validation dependence. ProtoNeX clusters models in behavior (prediction) space, elects complementary experts via multi-criteria selection, and fuses them into K compact prototypes. Instead of clustering data and training separate models per cluster, ProtoNeX clusters the models themselves and uses prototype fusion, promoting complementarity and behavioral diversity across the model space to enhance generalization.



<img width="12480" height="4065" alt="plot_Inner_View_Study" src="https://github.com/user-attachments/assets/68a7bd96-ebfd-4904-937b-83d2f7f76cba" />

Visualization of Train–Val Overfitting (TO) and Val–Test Overfitting (VO) gaps for two parent networks A and B and their genetically evolved child models. We investigated how
two overfitted models, A and B, evolve into
more generalizable child models through genetic operations.
To simulate the effect of overfitting, we deliberately trained A and B for extended epochs, exposing them
to both TO and VO divergence.
Then, apply weight-level crossover and mutation to generate
child networks. The genetic crossover process acts as a weight-space regularizer,
reinitializing over-optimized trajectories and promoting the
exploration of new parameter regions, reducing overfitting and improving generalization.







---

## Framework Structure

```
GeNeX/
├── GenE/                     # Generates a diverse population of networks via gradient-based training and genetic weight fusion mechanisms.
├── ProtoNeX/                 # Clusters the generated population, elects experts per cluster, and fuzes prototypes in weight-level.
```

---

## How to Run

1. Open GenE/CONFIGS.py -->
```bash
GENERAL_CONFIG = {

    "DEBUG": 1, # <----- If want to debug (very fast run), then put here: 1 // else for full put: 0

    "ROOT_PATH" :'../data/DISTRIBUTIONAL_DRIFTED', # <-----  replace < data > with the name of root datafolder, e.g. SKIN_CANCER
    # Due to size restrictions in GitHub download from:
    # LINK:
    # https://drive.google.com/file/d/1RcZsiCOuFFWXuEoiVptNvP7lDTGWlhvh/view?usp=sharing

    "STORAGE_PATH" :'../_models_storage', # your output storage folder path / this will be used next for ProtoNeX
    "train_dir" :'train',
    "val_dir" :'val',
    "test_dir" :'test',
    }

# left values are for full run | right values for fast run (debuging if was set with 1)
GenE_CONFIG = {
    "SIZE": [224, 64][GENERAL_CONFIG["DEBUG"]], # images size // good results also the 144 size.
    "N": [50, 10][GENERAL_CONFIG["DEBUG"]], # number of generated models / size of output model pool M
    "G": [5, 2][GENERAL_CONFIG["DEBUG"]], # number of total generations
    }
sel = int (GenE_CONFIG['N']/GenE_CONFIG['G']) # number of geneticaly evolved selected networks per generation --> which are progressively stored into the M output folder
N_g = [ GenE_CONFIG['N'],  GenE_CONFIG['N'] // 2 ][GENERAL_CONFIG["DEBUG"]] # number of randomly selected parent pairs
```
and specify configs.

2. Run:
```bash 
GenE/main.py
```
This will progressively update and store the evolved population in:
```bash 
 _models_storage/M
```
These networks will be used as input into the ProtoNeX module.

3. When GenE terminates (needs time), then just run:
```bash 
ProtoNeX/main.py
```

4. When ProtoNex terminates (fast compared to GenE),
```bash 
prototypes, weights = protonex.run()
```
it will produce the prototypes (a small number of the final specialized networks to the given task) and their corresponding combination weights stored in:
```bash 
ProtoNeX/ProtoNeX_output
```
These, form the final ensemble predictor (Inference) for the given task.

5. Finally, run inference:
```bash 
ProtoNeX/inference.py
```
in new unseen data.



In https://github.com/EmmanuelPintelas/-Shifted-Dataset-Creator we provide a tool for creating shifted-variants of a given dataset for simulating realistic/challenging distributional shifted environments for benchmarking ML models' robustness ability in any given task.
