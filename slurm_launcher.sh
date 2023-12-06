singularity exec --nv /home/jtyo/containers/deep_person_reid.sif bash -c '
    # Load conda and activate the environment
    . /opt/conda/etc/profile.d/conda.sh
    conda activate base

    # Change to the appropriate directory
    cd /home/jtyo/Repos/deep-person-reid

    # Start a run as a wandb agent
    wandb agent --count 1 z5knho2e
'