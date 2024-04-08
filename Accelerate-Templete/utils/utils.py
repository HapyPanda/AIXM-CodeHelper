import shutil
import os

def clean_checkpoint(folder,limit,logger):
    checkpoints = os.listdir(folder)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= limit:
        num_to_remove = len(checkpoints) - limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(folder, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)