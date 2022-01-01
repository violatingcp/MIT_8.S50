## Instruction on setting up appropriate software environment for project 2

The project is done in Jupyter Notebooks. It's an interactive IDE that is used in many scientific data anlysis projects! For example, in the annoucement of the lastest muon g-2 measurement, the spokeperson showed how they obtained the final measurement by sharing a video of them running a code cell in jupyter notebook: https://www.youtube.com/watch?v=81PfYnpuOPA (at 54:02).

If you are not very familiar with it, you can check out the website here: https://jupyter.org

To set up the coding environment, you usually need anaconda installed on your machine (it's not neccessary, but it makes managing different versions of software packages much easier): https://docs.anaconda.com/anaconda/install/index.html

After downloading anaconda, you can use an environment file (`environment.yml`) I created to set up all the packages at once using these commands:

Create conda environment:

```
conda-env create -f environment.yml
```

Activate the environment:

```
conda activate ERegress
```

And then do whatever you want in this environment (edit files, open notebooks, etc.). To deactivate the environment:

```
conda deactivate
```

If you make any update for the environment, please edit the `environment.yml` file and run:

```
conda env update --file environment.yml  --prune
```

Reference on conda environment here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


