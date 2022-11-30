# functional_atlas_explorer
Functional Atlas Explorer Project for Brainhack Western 2022

## Installation
1. Clone this repository.
   ```git clone https://github.com/carobellum/functional_atlas_explorer.git```

2. Navigate to the repository (```cd /path/to/your/repository/cone/functional_atlas_explorer```) and make a new environment.
   ```python -m venv env```

3. Activate your new environment.
   ```source env/bin/activate```

4. Install the packages that we need for this project.
   ```pip intall pandas numpy scipy nibabel SUITPy matplotlib seaborn pickle5 ipykernel neuroimagingtools dash ipykernel torch```

5. You might have to update your nbformat package (this might only be the case for some of you), so to be sure run ```pip install --upgrade nbformat```

Done! Now try running the cells in brainhack_example.ipynb and explore the plots. Try to understand what the code is doing and what you see in the plots. On Day 2, we are going to work on making integrative plots of the functional profiles of the voxels!

Let me know if you ran into any errors and we will fix them together :)

## Links


Have a look at the [cerebellar atlas viewer](https://www.diedrichsenlab.org/imaging/AtlasViewer/index.htm) to see an example of an interactive visualisation tool.

We will be working with [plotly](https://pypi.org/project/plotly/) to get our interactive functional atlas explorer running.
Check out [this chapter](https://dash.plotly.com/interactive-graphing) of the plotly guide that covers interactive graphing.
