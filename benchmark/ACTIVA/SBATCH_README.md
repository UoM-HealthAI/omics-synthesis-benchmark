# ACTIVA Training SBATCH Script Usage

## Basic Usage

Submit a job with default parameters:
```bash
sbatch train_activa.sbatch
```

## Custom Parameters

The script accepts positional arguments:
```bash
sbatch train_activa.sbatch [DATASET] [LR] [EPOCHS] [BATCH_SIZE] [OUTPUT_DIR]
```

### Examples

1. **Train on Tabula Muris with default settings:**
```bash
sbatch train_activa.sbatch tabula_muris
```

2. **Train with custom learning rate and epochs:**
```bash
sbatch train_activa.sbatch tabula_muris 0.0001 1000
```

3. **Train with all custom parameters:**
```bash
sbatch train_activa.sbatch covid 0.0002 500 64 ./covid_output/
```

## Parameters

- **DATASET**: Dataset name (default: "tabula_muris")
  - Options: "pbmc", "20k brain", "covid", "tabula_muris"
- **LR**: Learning rate (default: 0.0002)
- **EPOCHS**: Number of epochs (default: 500)
- **BATCH_SIZE**: Batch size (default: 128)
- **OUTPUT_DIR**: Output directory (default: "./outputs/${DATASET}_${SLURM_JOB_ID}/")

## Output

- **Logs**: Saved in `outputs/activa_${JOB_ID}.out` and `outputs/activa_${JOB_ID}.err`
- **Models**: Saved in the specified output directory
- **TensorBoard**: Logs saved in the output directory for visualization

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View real-time output:
```bash
tail -f outputs/activa_${JOB_ID}.out
```

## Customization

To modify for your cluster:
1. Adjust resource requirements (#SBATCH directives)
2. Uncomment and modify module loads
3. Uncomment and modify conda activation
4. Add any cluster-specific environment settings