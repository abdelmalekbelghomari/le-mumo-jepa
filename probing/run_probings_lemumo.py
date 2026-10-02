import subprocess
import re
import os
import argparse
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_single_free_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_memory = []
        for idx, line in enumerate(lines):
            used, total = map(int, line.split(','))
            free = total - used
            gpu_memory.append((free, idx))
        gpu_memory.sort(key=lambda x: x[0], reverse=True)
        return f"cuda:{gpu_memory[0][1]}"
    except:
        return "cuda:0"

def main():
    parser = argparse.ArgumentParser(description="Wrapper séquentiel x runs — probing Le MuMo JEPA")
    parser.add_argument('--amount_runs', type=int, default=10)
    parser.add_argument('--jepa_checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_small')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--ir_mean', type=str, default='0.449')
    parser.add_argument('--ir_std', type=str, default='0.226')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--csv_output', type=str, default=None) 
    parser.add_argument('--flir_root', type=str, default='/dev/shm/FLIR')
    parser.add_argument('--log_dir', type=str, default='./logs_linear_probing')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--tokenizer_by_modality', type=str, default='True',
                        help="Le MuMo : True (IR 1 canal)")
    parser.add_argument('--modes', nargs='+', default=['rgb', 'ir', 'both'],
                        choices=['rgb', 'ir', 'both', 'joint'])
    args = parser.parse_args()

    checkpoint_parent_dir = os.path.dirname(args.jepa_checkpoint)
    if args.output_dir is None:
        args.output_dir = checkpoint_parent_dir
    if args.csv_output is None:
        args.csv_output = os.path.join(checkpoint_parent_dir, f'bilan_probing_{args.amount_runs}_runs.csv')
    target_device = args.device if args.device is not None else get_single_free_gpu()

    print(f"📁 Checkpoint : {args.jepa_checkpoint}\n📁 Output : {args.output_dir}\n📊 CSV : {args.csv_output}\n🚀 Device : {target_device}\n🔀 tokenizer_by_modality : {args.tokenizer_by_modality} | IR norm : mean={args.ir_mean} std={args.ir_std}\n" + "-"*60)

    report_modes = [m.upper() for m in args.modes]
    metrics = {
        mode: {'mAP_global': [], 'mAP_person': [], 'mAP_car': [], 'mAP_bicycle': [], 'AUC_person': [], 'AUC_car': [], 'AUC_bicycle': []}
        for mode in report_modes
    }
    csv_rows = []

    for run in range(1, args.amount_runs + 1):
        seed = 42 + run
        print(f"▶️  RUN {run}/{args.amount_runs} | {target_device} | Seed: {seed}")
        cmd = [
            "python3", os.path.join(SCRIPT_DIR, "lemumo_linear_probe_per_tkn.py"),
            "--jepa_checkpoint", args.jepa_checkpoint,
            "--model_name", args.model_name,
            "--flir_root", args.flir_root,
            "--method", "features",
            "--epochs", "30",
            "--batch", "128",
            "--crop_size", str(args.crop_size),
            "--lr", "1e-3",
            "--output_dir", args.output_dir,
            "--log_dir", args.log_dir,
            "--use_saved_features", "True",
            "--workers", "0",
            "--seed", str(seed),
            "--device", target_device,
            "--patch_size", str(args.patch_size),
            "--tokenizer_by_modality", args.tokenizer_by_modality,
            "--ir_mean", args.ir_mean,
            "--ir_std", args.ir_std,
            "--modes", *args.modes,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            print(output)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Le RUN {run} a planté !")
            print(f"Code de sortie : {e.returncode}")
            print("--- ERREUR (stderr) ---")
            print(e.stderr) 
            print("-----------------------")
            raise e

        for mode in report_modes:
            run_data = {'run': run, 'seed': seed, 'modality': mode}
            map_match = re.search(fr"{mode}\s*│.*│\s*mAP=([\d\.]+)", output)
            mAP_val = float(map_match.group(1)) if map_match else np.nan
            metrics[mode]['mAP_global'].append(mAP_val)
            run_data['mAP_global'] = mAP_val

            epoch_pattern = fr"\[{mode}\]\s*30/30\s*\|.*\|.*person=([\d\.]+)\s*\|\s*car=([\d\.]+)\s*\|\s*bicycle=([\d\.]+)"
            epoch_match = re.search(epoch_pattern, output)
            for cls_idx, cls_name in enumerate(['person', 'car', 'bicycle']):
                cls_map = float(epoch_match.group(cls_idx + 1)) if epoch_match else np.nan
                metrics[mode][f'mAP_{cls_name}'].append(cls_map)
                run_data[f'mAP_{cls_name}'] = cls_map

            for cls_name in ['person', 'car', 'bicycle']:
                auc_match = re.search(fr"METRIC_AUC_{mode}_{cls_name.upper()}:\s*([\d\.]+)", output)
                cls_auc = float(auc_match.group(1)) if auc_match else np.nan
                metrics[mode][f'AUC_{cls_name}'].append(cls_auc)
                run_data[f'AUC_{cls_name}'] = cls_auc
            csv_rows.append(run_data)

    csv_headers = ['run', 'seed', 'modality', 'mAP_global', 'mAP_person', 'mAP_car', 'mAP_bicycle', 'AUC_person', 'AUC_car', 'AUC_bicycle']
    try:
        with open(args.csv_output, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n💾 Sauvegardé : {args.csv_output}")
    except Exception as e:
        print(f"⚠️ Erreur CSV : {e}")

    print("\n" + "="*55 + "\n 📊 RAPPORT STATISTIQUE FINAL\n" + "="*55)
    for mode in report_modes:
        print(f"\n🔹 MODE : {mode}\n" + "-"*50)
        for key, values in metrics[mode].items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                print(f"  {key:<12} │ Moyenne = {np.mean(valid_values):.4f} │ Écart-type (±) = {np.std(valid_values):.4f} │ Variance = {np.var(valid_values):.6f}")
    print("="*55)

if __name__ == '__main__':
    main()