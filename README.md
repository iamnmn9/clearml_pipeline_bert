# Bittensor Pipeline
Repository containing all experiments, tasks, pipeline, and other scripts for Bittensor

## Virtual Enviroment
```
# created venv 
python3 -m venv .venv 

# activate venv 
source .venv/bin/activate 

# install deps
python install -r requirements.txt
```

## Random Notes to be addressed later 
- you can specify direct execution queue in a `.pipeline` or `.component` with `pipeline_execution_queue=""`
- each clearml agent machine needs access to a github user that can reach the repository `clear-ml` user has been setup for this 
- clearml agent needs to have `virtualenv` install via pip3 to run 
- when editing files you need to push your latest commit to github and then run 
- clearml auto pulls from the requirments.txt in the repo
