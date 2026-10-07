import json
import matplotlib.pyplot as plt

def plot_loss_curve(log_file='loss_history_colab.json'):
    try:
        # 1. Load the recorded history
        with open(log_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{log_file}'. Make sure you've run train.py first.")
        return
        
    iters = history["iters"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # 2. Setup the canvas
    plt.figure(figsize=(10, 6))
    
    # 3. Plot the data lines
    plt.plot(iters, train_loss, label='Train Loss', color='#2563eb', linewidth=2)
    plt.plot(iters, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    
    # 4. Format the graph to look professional
    plt.title('algo-lm Training Progression', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    
    # Add a subtle grid for readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Remove the top and right borders for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 5. Save the image and display it
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    print("Success! Graph saved to your folder as 'loss_curve.png'")
    
    # Opens a window showing the graph (close the window to end the script)
    plt.show()

if __name__ == '__main__':
    plot_loss_curve()