import pickle
from pathlib import Path



def mr_pickler(variable, variable_name):
    """
    Pickle the variable for deeper inspeciton
    Args: variable: What you want to pickle
          variable_name: the string version of the varible to use as name
    Returns: None.  Output saved to file
    """
    my_dir = Path(__file__).parent / "pickle"
    my_dir.mkdir(exist_ok=True)
    filepath = Path(my_dir) / f'{variable_name}.pkl'

    with open(filepath, "wb") as pickle_file:
        pickle.dump(variable, pickle_file)
    
    print(f'{variable_name} pickled to {filepath}')