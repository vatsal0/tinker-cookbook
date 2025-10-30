import tinker
from tinker_cookbook import checkpoint_utils
log_path = '/nfshomes/vatsalb/tinker-cookbook/checkpoints/encoded_235b_mult'
out_path = '/nfshomes/vatsalb/tinker-cookbook/checkpoints/encoded_235b_mult_conv' # mkdir

service_client = tinker.ServiceClient()

async def main():
    checkpoints = checkpoint_utils.load_checkpoints_file(log_path)
    for checkpoint in checkpoints:
        print(checkpoint)
        print('getting client...')
        training_client = await service_client.create_training_client_from_state_async(
            checkpoint["state_path"]
        )
        batch_idx = checkpoint["batch"]
        print('saving checkpoint...')
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{batch_idx:06d}",
            log_path=out_path,
            kind="both",
            loop_state={"batch": batch_idx},
        )
        print('done')

import asyncio
asyncio.run(main())