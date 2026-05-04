import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram(output_path="assets/system_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    plt.text(50, 95, "GridPulse India: TSA System Architecture", 
             fontsize=20, weight="bold", ha="center", va="center", color="#303F9F")

    def draw_box(ax, x, y, width, height, title, lines):
        rect = patches.FancyBboxPatch(
            (x, y), width, height, 
            boxstyle="round,pad=1,rounding_size=2", 
            linewidth=2, edgecolor="#3F51B5", facecolor="#E8EAF6"
        )
        ax.add_patch(rect)
        plt.text(x + width/2, y + height - 3, title, 
                 fontsize=14, weight="bold", ha="center", va="center", color="#1A237E")
        
        for i, line in enumerate(lines):
            plt.text(x + 2, y + height - 8 - i*5, f"• {line}", 
                     fontsize=11, ha="left", va="center", color="#000000")

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#FF4081", lw=2, mutation_scale=20))

    # Boxes
    draw_box(ax, 5, 65, 25, 20, "1. Data Layer", 
             ["Synthetic demand data", "Hierarchy:", "  India → Maharashtra → Mumbai"])
    
    draw_box(ax, 38, 65, 25, 20, "2. Feature Engineering", 
             ["Fourier Seasonality", "Calendar Features", "(Month, Day of Week, Weekend)"])
    
    draw_box(ax, 70, 65, 25, 20, "3. Global LSTM Model", 
             ["Shared temporal representation", "Non-linear relationships", "Multi-region forecasting"])
    
    draw_box(ax, 20, 30, 25, 20, "4. Hierarchical Reconciliation", 
             ["sktime BottomUp approach", "Aggregates Mumbai → State → National", "Ensures consistent forecasts"])
    
    draw_box(ax, 55, 30, 25, 20, "5. Delivery & Reporting", 
             ["Forecast CSV generation", "GridPulse PPT generation", "Performance tracking"])

    # Bottom context box
    rect = patches.FancyBboxPatch(
        (10, 5), 80, 15, 
        boxstyle="round,pad=1,rounding_size=2", 
        linewidth=1, edgecolor="#4CAF50", facecolor="#E8F5E9"
    )
    ax.add_patch(rect)
    plt.text(50, 16, "Time Series Analysis (TSA) Application Context", 
             fontsize=14, weight="bold", ha="center", va="center", color="#2E7D32")
    plt.text(50, 11, "Fourier Terms for recurring seasonality cycles • Global Neural Networks for shared patterns", 
             fontsize=12, ha="center", va="center", color="#1B5E20")
    plt.text(50, 7, "• Hierarchical Time Series (HTS) Reconciliation for aggregation consistency •", 
             fontsize=12, ha="center", va="center", color="#1B5E20")

    # Arrows
    draw_arrow(ax, 30, 75, 38, 75)
    draw_arrow(ax, 63, 75, 70, 75)
    draw_arrow(ax, 82, 65, 32, 50)
    draw_arrow(ax, 45, 40, 55, 40)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
