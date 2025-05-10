#!/bin/bash
#SBATCH --job-name=decogen_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=a5000
#SBATCH --nodelist=compute-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=19G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=thomas.fredrich@student.uliege.be
#SBATCH --mail-type=BEGIN,END,FAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Afficher des informations sur le travail
echo "Date de début: $(date)"
echo "Nœud: $(hostname)"
echo "Répertoire de travail: $(pwd)"
echo "ID du Job: $SLURM_JOB_ID"

# Charger l'environnement conda
source ~/.bashrc
conda activate decogen || echo "Échec de l'activation de l'environnement conda 'decogen'"

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p output/checkpoints output/samples

# Configuration de wandb (décommentez et configurez si nécessaire)
# export WANDB_API_KEY="votre_clé_api"
# export WANDB_ENTITY="votre_nom_utilisateur"
# export WANDB_PROJECT="decogen"

# Lancer l'entraînement
echo "Démarrage de l'entraînement..."
python main.py train --batch_size 2 --epochs 100

# Afficher la date de fin
echo "Date de fin: $(date)"