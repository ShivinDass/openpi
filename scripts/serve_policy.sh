# config=gdg_libero_single_pi0_lora
# exp=val_5_pi0_lora

config=gdg_libero_cotrain_pi0_lora
exp=val_5-val_5_human_curated_exact_data
# exp=val_5-task5traj5_traj20_explore_exploit_vlm_evol_soup_butter_tray_1_steps10k

uv run scripts/serve_policy.py --env LIBERO --port 8001 \
            policy:checkpoint --policy.config $config \
            --policy.dir checkpoints/${config}/${exp}/4999/