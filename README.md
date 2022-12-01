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
   ```pip intall pandas numpy scipy nibabel SUITPy matplotlib seaborn pickle5 ipykernel neuroimagingtools ipykernel torch dash jupyter-dash dash_boostrap_components```

5. You might have to update your nbformat package (this might only be the case for some of you), so to be sure run ```pip install --upgrade nbformat```

Done! Now try running the cells in the jupyter notebook ```brainhack_example.ipynb``` and explore the plots. Try to understand what the code is doing and what you see in the plots. On Day 2, we are going to work on making integrative plots of the functional profiles of the voxels!

Let me know if you ran into any errors and we will fix them together :)

### Troubleshooting
If you run into errors, there is a few things to check before looking further:
- Are you running your python code in the correct environment?
  - Did you activate the environment for your terminal only, but your jupyter notebook is not using your environment? Depending on which code editor you use (I prefer VSCode), you will have to look in different places. For VSCode, have a look [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
- Did you install all of the packages in step 4? Did you install them into the correct environment?

## Using dash

Have a look at the scripts starting with ```app```.

```app.py```   [Tutorial](https://dash.plotly.com/layout)

```app_hover.py``` [Tutorial](https://dash.plotly.com/interactive-graphing)

```app_tooltip.py``` [Tutorial](https://dash.plotly.com/dash-core-components/tooltip?_gl=1*1ljxuab*_ga*Mjk4OTgyNTMuMTY2OTIyNjI0Ng..*_ga_6G7EE0JNSC*MTY2OTg0OTU4MC44LjEuMTY2OTg1MDg1MC4wLjAuMA)

More Dash Tutorials can be found [here](https://medium.com/sfu-cspmp/plotly-dash-story-edbb8c3e151e).

These are example scripts taken from the [Plotly Dash Guide](https://dash.plotly.com/). Try playing around with these and understanding the code, particularly of ```app_tooltip.py``` - this is pretty close to what we want to build!




## Links

Have a look at the [cerebellar atlas viewer](https://www.diedrichsenlab.org/imaging/AtlasViewer/index.htm) to see an example of an interactive visualisation tool.

We will be working with [plotly](https://pypi.org/project/plotly/) to get our interactive functional atlas explorer running.
Check out [this chapter](https://dash.plotly.com/interactive-graphing) of the plotly guide that covers interactive graphing.
