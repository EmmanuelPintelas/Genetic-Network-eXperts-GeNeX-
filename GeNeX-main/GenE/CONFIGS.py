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
    "SIZE": [224, 64][GENERAL_CONFIG["DEBUG"]], # images size // good results also with the 144 size.
    "N": [50, 10][GENERAL_CONFIG["DEBUG"]], # number of generated models / size of output model pool M
    "G": [5, 2][GENERAL_CONFIG["DEBUG"]], # number of total generations
    }
sel = int (GenE_CONFIG['N']/GenE_CONFIG['G']) # number of geneticaly evolved selected networks per generation --> which are progressively stored into the M output folder
N_g = [ GenE_CONFIG['N'],  GenE_CONFIG['N'] // 2 ][GENERAL_CONFIG["DEBUG"]] # number of randomly selected parent pairs