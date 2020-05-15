import matplotlib.pyplot as plt
import os

def draw_graph():

    ratios = ["50,000", "12,500", "6,250", "3,125", "1,562"]
    plt.figure(figsize=(8, 4))
    train_acc = [75.512, 87.016, 96.992, 100, 100]
    test_acc = [66.86, 60.24, 48.55, 35.03, 33.040000000000006]
    plt.plot(ratios, train_acc, "r", linewidth=1, label="train acc")
    plt.plot(ratios, test_acc, "g", linewidth=1, label="test acc")
    plt.xlabel("Labeled Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("save_data", "res.jpg"))

draw_graph()