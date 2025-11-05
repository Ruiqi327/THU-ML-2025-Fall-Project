import random
import numpy as np
import os
import argparse

def generate_dataset_linear(type='linear', num_samples=1000, num_features=4, output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = np.random.rand(num_samples, num_features)
    X[:,0]= np.floor(X[:,0]*8+1)  # Scale learning time to [1,8]
    X[:,1]= np.floor(X[:,1]*70+30)  # Scale midterm grade to [30,100]
    X[:,2]= np.floor(X[:,2]*70+30)  # Scale attendance rate to [30,100]
    X[:,3]= np.floor(X[:,3]*70+30)  # Scale homework completion to [30,100]

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)

    coefficients = np.array([0.5, 0.2, 0.2, 0.1])

    y = X_norm.dot(coefficients)
    y_label = np.zeros(num_samples)

    if type == 'linear':
        filename = 'linear_dataset.csv'
        for i in range(num_samples):
            y_label[i] = 1 if 0.5 < y[i] else -1
    elif type == 'nonlinear':
        filename = 'nonlinear_dataset.csv'
        for i in range(num_samples):
            y_label[i] = 1 if 0.25 < y[i] < 0.75 else -1
    else:
        filename = 'noisy_dataset.csv'
        for i in range(num_samples):
            y_label[i] = 1 if 0.5 + random.uniform(-0.3, 0.3) < y[i] else -1
    
    y_label_reshaped = y_label.reshape(-1, 1)
    combined_dataset = np.hstack((X, y_label_reshaped))

    output_path = os.path.join(output_dir, filename)
    headers = "learning_time,midterm_grade,attendance_rate,homework_completion,pass_fail"
    np.savetxt(output_path, combined_dataset, delimiter=',', header=headers, comments='', fmt='%d')

def main():
    parser = argparse.ArgumentParser(description="4 feature dataset")
    parser.add_argument("--output_dir", type=str, help="Directory containing dataset files")
    parser.add_argument('--data_scale', type=float, default=1000, help='data scale (number of samples)')
    args = parser.parse_args()
    output_dir=args.output_dir
    num_samples=int(args.data_scale)
    num_features = 4
    generate_dataset_linear(type='linear', num_samples=num_samples, num_features=num_features, output_dir=output_dir)
    generate_dataset_linear(type='nonlinear', num_samples=num_samples, num_features=num_features, output_dir=output_dir)
    generate_dataset_linear(type='noisy', num_samples=num_samples, num_features=num_features, output_dir=output_dir)

if __name__ == "__main__":
    main()