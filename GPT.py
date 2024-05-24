import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def task1GPT():
    A = diabetes.data
    b = diabetes.target

    # Gradient Descent parameters
    eps = 0.001  # Step size (learning rate)
    delta = 1e-6  # Stopping condition
    max_iterations = 2000  # Maximum number of iterations

    # Initialize variables
    n_samples, n_features = A.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    error_plot = []  # To store error values

    # Define the error function
    def compute_error(A, x, b):
        return np.linalg.norm(A @ x - b) ** 2

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A.T @ A @ x - A.T @ b)  # Gradient of the least squares cost function
        x = x - eps * grad  # Update the parameters
        error = compute_error(A, x, b)  # Compute the error
        error_plot.append(error)  # Store the error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            print(f'Gradient Descent converged after {i} iterations')
            break

    # If gradient descent did not converge within the maximum number of iterations
    if i == max_iterations - 1:
        print('Gradient Descent did not converge within the maximum number of iterations')

    # Final optimized x
    print('Optimized x:', x)

    # Plotting the error over iterations
    plt.plot(error_plot)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Error Plot of Gradient Descent')
    plt.show()


def task2GPT(max_iterations=100000, eps=0.001, delta=1e-6):
    # Load diabetes dataset
    diabetes = load_diabetes()
    A = diabetes.data  # Feature matrix (442 x 10)
    b = diabetes.target  # Target vector (442,)

    # Split the dataset into training and test sets
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

    # Gradient Descent parameters
    # eps = 0.001  # Step size (learning rate)
    # delta = 1e-6  # Stopping condition
    # max_iterations = 1000  # Maximum number of iterations

    # Initialize variables
    n_samples, n_features = A_train.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    train_error_plot = []  # To store training error values
    test_error_plot = []  # To store test error values

    # Define the error function
    def compute_error(A, x, b):
        return np.linalg.norm(A @ x - b) ** 2

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A_train.T @ A_train @ x - A_train.T @ b_train)  # Gradient of the least squares cost function
        x = x - eps * grad  # Update the parameters

        train_error = compute_error(A_train, x, b_train)  # Compute the training error
        test_error = compute_error(A_test, x, b_test)  # Compute the test error

        train_error_plot.append(train_error)  # Store the training error
        test_error_plot.append(test_error)  # Store the test error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            print(f'Gradient Descent converged after {i} iterations')
            break

    # If gradient descent did not converge within the maximum number of iterations
    if i == max_iterations - 1:
        print('Gradient Descent did not converge within the maximum number of iterations')

    # Final optimized x
    print('Optimized x:', x)

    # Plotting the train and test errors over iterations
    plt.plot(train_error_plot, label='Train Error')
    plt.plot(test_error_plot, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Train and Test Error Plot of Gradient Descent')
    plt.legend()
    plt.show()


def compute_error(A, x, b, alpha=0):
    return np.linalg.norm(A @ x - b) ** 2 + alpha * np.linalg.norm(x) ** 2


def gradient_descent_ridge(A_train, b_train, A_test, b_test, eps=0.001, delta=1e-6, max_iterations=1000, alpha=1.0):
    n_samples, n_features = A_train.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    train_error_plot = []  # To store training error values
    test_error_plot = []  # To store test error values

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A_train.T @ (A_train @ x - b_train) + alpha * x)  # Gradient with regularization
        x = x - eps * grad  # Update the parameters

        train_error = compute_error(A_train, x, b_train, alpha)  # Compute the training error
        test_error = compute_error(A_test, x, b_test, alpha)  # Compute the test error

        train_error_plot.append(train_error)  # Store the training error
        test_error_plot.append(test_error)  # Store the test error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            break

    return train_error_plot, test_error_plot


def task3GPT():
    # Load diabetes dataset
    A = diabetes.data  # Feature matrix (442 x 10)
    b = diabetes.target  # Target vector (442,)

    num_runs = 10
    max_iterations = 1000
    eps = 0.001
    delta = 1e-6
    alpha = 1.0

    train_errors = np.zeros((num_runs, max_iterations))
    test_errors = np.zeros((num_runs, max_iterations))

    for run in range(num_runs):
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=run)

        train_error_plot, test_error_plot = gradient_descent_ridge(A_train, b_train, A_test, b_test, eps, delta,
                                                                   max_iterations, alpha)

        # Pad with the last error if the iteration count is less than max_iterations
        train_errors[run, :len(train_error_plot)] = train_error_plot
        train_errors[run, len(train_error_plot):] = train_error_plot[-1]
        test_errors[run, :len(test_error_plot)] = test_error_plot
        test_errors[run, len(test_error_plot):] = test_error_plot[-1]

    # Compute average and minimum errors
    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)
    min_train_errors = np.min(train_errors, axis=0)
    min_test_errors = np.min(test_errors, axis=0)

    # Plotting average errors
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_errors, label='Average Train Error')
    plt.plot(avg_test_errors, label='Average Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Average Train and Test Error Plot')
    plt.legend()
    plt.show()

    # Plotting minimum errors
    plt.figure(figsize=(10, 5))
    plt.plot(min_train_errors, label='Minimum Train Error')
    plt.plot(min_test_errors, label='Minimum Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Minimum Train and Test Error Plot')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    diabetes = load_diabetes()
    print("Data Shape:", diabetes.data.shape)  # (442, 10)
    print("Labels Shape:", diabetes.target.shape)  # (442, )

    task1GPT()
    task2GPT(20000,1e-4, 1e-15)
    task3GPT()
