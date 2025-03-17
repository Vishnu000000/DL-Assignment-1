# visualize.py
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_confusion_matrix():
    wandb.login()
    api = wandb.Api()
    project_path = "cs24m022-iit-madras-foundation/Deep_Learning_Assignment1_cs24m022"
    
    try:
        # Retrieve runs in the project.
        runs = list(api.runs(project_path))
        # Filter runs belonging to the sweep "vhyp7it5"
        sweep_runs = [run for run in runs if run.sweep_id == "vhyp7it5"]
        image_artifacts = []
        for run in sweep_runs:
            try:
                arts = list(run.logged_artifacts())
            except Exception:
                arts = list(run.artifacts())
            for art in arts:
                if art.type == "image" and "confusion_matrix" in art.name.lower():
                    image_artifacts.append(art)
        if not image_artifacts:
            print("No confusion matrix artifacts found for sweep 'vhyp7it5' in project:", project_path)
            return
        best_artifact = max(image_artifacts, key=lambda art: art.metadata.get("val_accuracy", 0))
        print("Best validation accuracy:", best_artifact.metadata.get("val_accuracy"))
        best_artifact_dir = best_artifact.download()
        img_path = os.path.join(best_artifact_dir, "confusion_matrix.png")
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Best Confusion Matrix\nVal Accuracy: {best_artifact.metadata.get('val_accuracy', 'N/A'):.2f}%", 
                     pad=20, fontsize=16, color='navy')
        plt.tight_layout()
        plt.savefig("final_confusion_matrix.png", bbox_inches='tight', dpi=300)
        plt.show()
        run = wandb.init(project=project_path, name="Final_CM_Visualization", job_type="analysis")
        wandb.log({"Final Confusion Matrix": wandb.Image("final_confusion_matrix.png")})
        wandb.finish()
    except Exception as e:
        print("An error occurred during visualization:", e)

if __name__ == "__main__":
    visualize_confusion_matrix()
