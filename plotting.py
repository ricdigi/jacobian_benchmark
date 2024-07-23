import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename='data/results_pendulum.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_performance(data):
    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Plot total times against input size
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='input_size', y='total_time', hue='implementation', marker='o')
    plt.title('Performance of Different Jacobian Implementations on the n_link_pendulum_on_cart model')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Total Time (s)')
    plt.legend(title='Implementation')
    plt.grid(True)
    plt.show()



data = load_data()
plot_performance(data)
