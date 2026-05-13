"""
Complete initialization, training, and evaluation script for the PI-GNN
surrogate model using EPytFlow.

Usage
-----
    python run.py --inp <path_to_inp> [options]

Examples
--------
    # Train from an .inp file (runs simulation internally)
    python run.py --inp data/Network-Net1_scenarios/scenario-Network-Net1_randDemand=True.inp

    # Use a pre-generated ScadaData file
    python run.py --inp data/Network-Net1_scenarios/scenario-Network-Net1_randDemand=True.inp \
                  --scada data/hanoi_randDemand=True_training.epytflow_scada_data

    # Evaluate a saved checkpoint
    python run.py --inp data/Network-Net1_scenarios/scenario-Network-Net1_randDemand=True.inp \
                  --eval-only --model-path tmp/pi_gnn_model.pt
"""

import argparse
import os
import json
import datetime
import shutil

import numpy as np
import torch

from pi_gnn_surrogate_epytflow import PIGNNModel, EPYTFLOW_NETWORKS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PI-GNN surrogate – train & evaluate")

    # --- Data ---
    p.add_argument("--inp", type=str, default=None,
                   help="Path to the EPANET .inp network file.")
    p.add_argument("--network", type=str, default="anytown",
                   help="Name of a built-in EPytFlow network to download "
                        f"(one of: {', '.join(sorted(EPYTFLOW_NETWORKS.keys()))}). "
                        "Overrides --inp.")
    p.add_argument("--scada", type=str, default=None,
                   help="Path to a .epytflow_scada_data file. "
                        "If omitted, a simulation is run from the .inp file.")
    p.add_argument("--sim_duration", type=int, default=120 * 24 * 3600,
                   help="Simulation duration in seconds (default: 120 days). "
                        "Only used when --scada is not provided.")
    p.add_argument("--hydraulic_dt", type=int, default=1800,
                   help="Hydraulic timestep in seconds (default: 1800).")
    p.add_argument("--randomize_demands", action="store_true", default=False,
                   help="Randomize demands in the simulation (default: False).")

    # --- Splits ---
    p.add_argument("--n_samples", type=int, default=(120 * 24 * 3600) // 1800,
                   help="Max timesteps to use (default: all).")

    # --- Model architecture ---
    p.add_argument("--M_l", type=int, default=128,
                   help="Latent dimension (default: 128).")
    p.add_argument("--I", type=int, default=5,
                   help="Number of GNN message-passing layers (default: 5).")

    # --- Training ---
    p.add_argument("--n_epochs", type=int, default=1500,
                   help="Training epochs (default: 1500).")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default: 1e-4).")
    p.add_argument("--batch_size", type=int, default=96,
                   help="Batch size (default: 96).")
    p.add_argument("--decay_step", type=int, default=150)
    p.add_argument("--decay_rate", type=float, default=0.75)
    p.add_argument("--grad_clip", type=float, default=1e-6)
    p.add_argument("--train_with_rand_demands", type=bool, default=True,
                   help="Whether to train with randomized demands (default: True).")
    p.add_argument("--dem_dist", type=str, default="uniform",
                   help="Demand noise distribution for augmentation (default: uniform).")
    p.add_argument("--dem_dist_fac", type=float, default=0.1,
                   help="Demand noise factor for augmentation.")
    p.add_argument("--train_with_rand_dias", type=bool, default=True,
                   help="Whether to train with randomized diameters (default: True).")
    p.add_argument("--dia_dist", type=str, default="uniform",
                   help="Diameter noise distribution for augmentation.")
    p.add_argument("--dia_dist_fac", type=float, default=0.025,
                   help="Diameter noise factor for augmentation.")

    # --- Checkpointing ---
    p.add_argument("--save_dir", type=str, default="tmp/" + str(datetime.date.today()),
                   help="Directory to save model and results. "
                        "Default: tmp/<date>/")
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to a saved model checkpoint to load.")

    # --- Modes ---
    p.add_argument("--eval_only", type=bool, default=False,
                   help="Skip training; only evaluate a saved checkpoint.")
    p.add_argument("--no_eval", type=bool, default=False,
                   help="Skip evaluation after training.")
    p.add_argument("--predict", type=bool, default=True,
                   help="Run prediction with SCADA data using a trained model.")
    p.add_argument("--compute_gradients", type=bool, default=True,
                   help="Compute gradients with SCADA data using a trained model.")

    cli_args = p.parse_args()

    if cli_args.inp is None and cli_args.network is None:
        p.error("Either --inp or --network is required")

    return cli_args


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    #  Save directory
    # ------------------------------------------------------------------
    save_dir = args.save_dir or os.path.join("tmp", str(datetime.date.today()))
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  1. Initialize model
    # ------------------------------------------------------------------
    if args.network is not None:
        print(f"\n{'='*60}")
        print(f"Downloading / loading built-in network: {args.network}")
        print(f"{'='*60}")
        model = PIGNNModel.from_network(args.network)
        args.inp = model.inp_file          # record resolved .inp path
        print(f"  .inp file: {model.inp_file}")
    else:
        print(f"\n{'='*60}")
        print(f"Initializing PIGNNModel with: {args.inp}")
        print(f"{'='*60}")
        model = PIGNNModel(inp_file=args.inp)

    # Save a permanent snapshot of the resolved .inp file for reproducibility.
    resolved_inp_path = os.path.abspath(model.inp_file)
    saved_inp_name = os.path.basename(resolved_inp_path).lower()
    saved_inp_path = os.path.join(save_dir, saved_inp_name)
    shutil.copy2(resolved_inp_path, saved_inp_path)
    args.inp = saved_inp_path
    model.inp_file = saved_inp_path
    print(f".inp snapshot saved to {saved_inp_path}")

    # Save run config for reproducibility using the persisted .inp path.
    config_path = os.path.join(save_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    model.batch_size = args.batch_size
    
    # ------------------------------------------------------------------
    #  Optional: Allowing for random sampling if the inp files contains
    #  only the base demands and no patterns. Moreover, this also allows
    #  for randomizing the diameters during training.
    # ------------------------------------------------------------------
    model.train_with_rand_demands = args.train_with_rand_demands    # Whether to apply random demand augmentation during training
    model.dem_dist = args.dem_dist                                  # Distribution type for demand augmentation (e.g., "uniform", "normal")
    model.dem_dist_fac = args.dem_dist_fac                          # Factor controlling the magnitude of demand augmentation (e.g., 0.5 means up to ±50% noise)  
    model.train_with_rand_dias = args.train_with_rand_dias          # Whether to apply random diameter augmentation during training
    model.dia_dist = args.dia_dist                                  # Distribution type for diameter augmentation (e.g., "uniform", "normal")
    model.dia_dist_fac = args.dia_dist_fac                          # Factor controlling the magnitude of diameter augmentation (e.g., 0.1 means up to ±10% noise)

    # ------------------------------------------------------------------
    #  2. Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")

    if args.scada is not None:
        print(f"  Source: ScadaData file  ({args.scada})")
        model.load_epytflow_scada(args.scada)
    else:
        print(f"  Source: EPytFlow simulation from .inp")
        print(f"  Duration: {args.sim_duration}s  |  dt: {args.hydraulic_dt}s  |  "
              f"Randomize demands: {args.randomize_demands}")
        model.load_from_inp(
            simulation_duration_sec=args.sim_duration,
            hydraulic_timestep=args.hydraulic_dt,
            randomize_demands=args.randomize_demands,
        )

    g = model._wdn_graph
    assert g is not None and g.X is not None
    T, N = g.X.shape[:2]
    n_edges = g.edge_attr.shape[1] // 2  # bidirectional
    print(f"  Timesteps: {T}  |  Nodes: {N}  |  Links: {n_edges}")
    print(f"  Reservoirs (idx): {g.reservoirs}")
    print(f"  Node names: {list(g.node_names)}")

    # ------------------------------------------------------------------
    #  3. Prepare train / val / test splits
    # ------------------------------------------------------------------
    print("\nPreparing data splits...")
    model.prepare_data(n_samples=args.n_samples)
    assert model._train_wds is not None and model._train_wds.X is not None
    assert model._val_wds is not None and model._val_wds.X is not None
    assert model._test_wds is not None and model._test_wds.X is not None
    print(f"  Train: {model._train_wds.X.shape[0]} steps  |  "
          f"Val: {model._val_wds.X.shape[0]} steps  |  "
          f"Test: {model._test_wds.X.shape[0]} steps")

    # ------------------------------------------------------------------
    #  4. Build model
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    model.build_model(
        M_l=args.M_l,
        I=args.I,
        n_epochs=args.n_epochs,
    )

    # Load checkpoint if provided
    if args.model_path is not None:
        print(f"  Loading checkpoint: {args.model_path}")
        model.load_model(args.model_path)

    # ------------------------------------------------------------------
    #  5. Train
    # ------------------------------------------------------------------
    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"Training for {args.n_epochs} epochs")
        print(f"  device={model.device}")
        print(f"  lr={args.lr}  batch_size={args.batch_size}  "
              f"decay_step={args.decay_step}  decay_rate={args.decay_rate}  "
              f"grad_clip={args.grad_clip}")
        print(f"{'='*60}\n")

        train_results = model.train(
            n_epochs=args.n_epochs,
            lr=args.lr,
            decay_step=args.decay_step,
            decay_rate=args.decay_rate,
            grad_clip=args.grad_clip,
            save_dir=save_dir,
            verbose=True,
        )

        print(f"\nModel saved to: {train_results['model_path']}")
        print(f"Final train loss: {train_results['train_losses'][-1]:.8f}")
        if train_results["val_losses"]:
            print(f"Final val loss:   {train_results['val_losses'][-1]:.8f}")

        # Save loss curves
        losses_path = os.path.join(save_dir, "losses.npz")
        np.savez(
            losses_path,
            train=np.array(train_results["train_losses"]),
            val=np.array(train_results["val_losses"]),
        )
        print(f"Loss curves saved to: {losses_path}")

    # ------------------------------------------------------------------
    #  6. Evaluate
    # ------------------------------------------------------------------
    if not args.no_eval:
        print(f"\n{'='*60}")
        print("Evaluating on test set")
        print(f"{'='*60}\n")

        eval_results = model.evaluate(batch_size=args.batch_size, model_path=args.model_path)

        heads_pred = eval_results["heads_pred"]    # [T_test, N, 1]
        heads_gt = eval_results["heads_gt"]        # [T_test, N, F_node]
        demands_pred = eval_results["demands_pred"]
        demands_gt = eval_results["demands_gt"]
        flows_pred = eval_results["flows_pred"]
        flows_gt = eval_results["flows_gt"]

        # Compute head MAE 
        head_mae = torch.abs(heads_pred - heads_gt).mean().item()
        head_mae_per_node = torch.abs(heads_pred - heads_gt).mean(dim=0).squeeze()

        print(f"  Head MAE (overall): {head_mae:.4f}")
        if g.node_names is not None:
            print(f"  Head MAE per node:")
            for i, name in enumerate(g.node_names):
                print(f"    {name}: {head_mae_per_node[i]:.4f}")

        # Demand MAE
        demand_mae = torch.abs(demands_pred - demands_gt).mean().item()
        print(f"  Demand MAE (overall): {demand_mae:.6f}")

        # Flow MAE
        flow_mae = None
        if flows_gt is not None:
            flow_mae = torch.abs(flows_pred - flows_gt).mean().item()
            print(f"  Flow MAE (overall): {flow_mae:.6f}")
        else:
            print("  Flow MAE (overall): N/A (no flow ground truth available)")

        # Save evaluation results
        eval_path = os.path.join(save_dir, "eval_results.npz")
        np.savez(
            eval_path,
            heads_pred=heads_pred.numpy(),
            heads_gt=heads_gt.numpy(),
            demands_pred=demands_pred.numpy(),
            demands_gt=demands_gt.numpy(),
            flows_pred=flows_pred.numpy(),
            flows_gt=flows_gt.numpy() if flows_gt is not None else np.array([]),
            test_losses=np.array(eval_results["test_losses"]),
            head_mae=head_mae,
            demand_mae=demand_mae,
            flow_mae=np.nan if flow_mae is None else flow_mae,
        )
        print(f"\nEvaluation results saved to: {eval_path}")

    print(f"\nDone. All outputs in: {save_dir}")

    # ------------------------------------------------------------------
    #  7. Prediction with the same or other SCADA data (optional)
    # ------------------------------------------------------------------
    if args.predict:
        print(f"\n{'='*60}")
        print("Predicting with SCADA data")
        print(f"{'='*60}\n")

        eval_results = model.predict(scada_data=args.scada, model_path=args.model_path)

        heads_pred = eval_results["heads_pred"]    # [T_test, N, 1]
        heads_gt = model._wdn_graph.X[..., 0:1]        # [T_test, N, 1]
        demands_pred = eval_results["demands_pred"].relu()
        demands_gt = model._wdn_graph.X[..., 1:2].relu()
        flows_pred = eval_results["flows_pred"]
        flows_gt = model._gt_flows_raw

        # Compute head MAE 
        head_mae = torch.abs(heads_pred - heads_gt).mean().item()
        head_mae_per_node = torch.abs(heads_pred - heads_gt).mean(dim=0).squeeze()

        print(f"  Head MAE (overall): {head_mae:.4f}")
        if g.node_names is not None:
            print(f"  Head MAE per node:")
            for i, name in enumerate(g.node_names):
                print(f"    {name}: {head_mae_per_node[i]:.4f}")

        # Demand MAE
        demand_mae = torch.abs(demands_pred - demands_gt).mean().item()
        print(f"  Demand MAE (overall): {demand_mae:.6f}")

        # Flow MAE
        flow_mae = None
        if flows_gt is not None:
            flow_mae = torch.abs(flows_pred - flows_gt).mean().item()
            print(f"  Flow MAE (overall): {flow_mae:.6f}")
        else:
            print("  Flow MAE (overall): N/A (no flow ground truth available)")

    # ------------------------------------------------------------------
    #  7. Prediction with the same or other SCADA data and gradient 
#         computation (optional)
    # ------------------------------------------------------------------
    if args.compute_gradients:
        print(f"\n{'='*60}")
        print("Predicting with SCADA data and computing gradients")
        print(f"{'='*60}\n")

        n_samples = 2  # Use a small number of samples for gradient computation to save time and memory
        eval_results = model.get_gradients(scada_data=args.scada, model_path=args.model_path,
                                            gradient_input="demands", gradient_output="heads",
                                            n_samples=n_samples)

        heads_pred = eval_results["heads_pred"]    # [T_test, N, 1]
        heads_gt = model._wdn_graph.X[:n_samples, :, 0:1]        # [T_test, N, 1]
        demands_pred = eval_results["demands_pred"].relu()
        demands_gt = model._wdn_graph.X[:n_samples, :, 1:2].relu()
        flows_pred = eval_results["flows_pred"]
        flows_gt = model._gt_flows_raw[:n_samples]
        gradients = eval_results["grads"]  # [T_test, N, 1]

        grad_path = os.path.join(save_dir, "gradients.npz")
        np.savez(
            grad_path,
            gradients=gradients.numpy(),
        )
        print(f"\nGradients saved to: {grad_path}")

        # Compute head MAE 
        head_mae = torch.abs(heads_pred - heads_gt).mean().item()
        head_mae_per_node = torch.abs(heads_pred - heads_gt).mean(dim=0).squeeze()

        print(f"  Head MAE (overall): {head_mae:.4f}")
        if g.node_names is not None:
            print(f"  Head MAE per node:")
            for i, name in enumerate(g.node_names):
                print(f"    {name}: {head_mae_per_node[i]:.4f}")

        # Demand MAE
        demand_mae = torch.abs(demands_pred - demands_gt).mean().item()
        print(f"  Demand MAE (overall): {demand_mae:.6f}")

        # Flow MAE
        flow_mae = None
        if flows_gt is not None:
            flow_mae = torch.abs(flows_pred - flows_gt).mean().item()
            print(f"  Flow MAE (overall): {flow_mae:.6f}")
        else:
            print("  Flow MAE (overall): N/A (no flow ground truth available)")





if __name__ == "__main__":
    main()
